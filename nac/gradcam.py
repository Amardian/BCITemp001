"""Grad-CAM with the softplus sub-gradient surrogate (Section 4.3.3).

Eq. (14):  alpha_k^c = (1/Z) sum_{i,j} dY^c / dA_ij^k        (GAP of grads)
Eq. (15):  L_GradCAM^c = ReLU( sum_k alpha_k^c A^k )
Eq. (11):  d ReLU(x)/dx  ~  sigmoid(x / beta_soft) / beta_soft

The channel saliency S(c) (used by ROI_frac / SDS / L_SAL / CSI / SPR) is
the Grad-CAM mass of EEG channel c:
    S(c) = sum_{h,w} ReLU( sum_k alpha_k^c A^k[c, h, w] )

`subgrad="softplus"` keeps the exact ReLU forward and swaps the backward
for the smooth everywhere-positive surrogate of Eq. (11), which is what
makes SPKD's saliency-alignment term trainable (Proposition 4.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SoftplusReLU(torch.autograd.Function):
    """Forward: exact ReLU. Backward: softplus sub-gradient (Eq. 11)."""

    @staticmethod
    def forward(ctx, x, beta_soft):
        ctx.save_for_backward(x)
        ctx.beta_soft = beta_soft
        return F.relu(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        beta = ctx.beta_soft
        surrogate = torch.sigmoid(x / beta) / beta
        return grad_out * surrogate, None


def softplus_relu(x, beta_soft=1.0, subgrad="softplus"):
    if subgrad == "softplus" and torch.is_grad_enabled() and x.requires_grad:
        return _SoftplusReLU.apply(x, beta_soft)
    return F.relu(x)


def grab_features(model, x):
    """Forward pass returning the last-conv feature maps (B, C, K, f, t)."""
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    raise TypeError("model must expose forward_features()")


@torch.enable_grad()
def channel_saliency(model, x, class_idx=None, subgrad="relu",
                     beta_soft=1.0, create_graph=False, detach=True,
                     normalize=True, layer=-1):
    """Grad-CAM channel saliency S(c) for a batch of trials.

    Args:
      model      : CNNGRU (or pruned/quantized variant)
      x          : (B, C, F, T) CWT features
      class_idx  : (B,) class whose score Y^c is explained (default: true
                   class not knowable here -> predicted class argmax)
      subgrad    : 'relu' (standard) or 'softplus' (Eq. 11 surrogate)
      create_graph: True makes S differentiable w.r.t. model params
                   (used by SPKD); requires subgrad='softplus'
      layer      : tower block index for Grad-CAM (-1 = last conv layer)

    Returns:
      S : (B, C) channel saliency (L1-normalized to sum 1 if normalize)
    """
    was_training = model.training
    model.eval()
    feats = grab_features(model, x)              # (B, C, K, f, t) w/ grad
    logits = model.forward_head(feats)           # same graph -> single pass
    if class_idx is None:
        class_idx = logits.argmax(dim=1)
    score = logits.gather(1, class_idx.view(-1, 1)).sum()

    grads = torch.autograd.grad(score, feats, create_graph=create_graph)[0]

    # alpha_k = GAP over every map position (channel and spatial)  Eq. 14
    alpha = grads.mean(dim=(1, 3, 4))             # (B, K)
    pre = (feats * alpha[:, None, :, None, None]).sum(dim=2)   # (B, C, f, t)
    maps = softplus_relu(pre, beta_soft, subgrad=subgrad)
    S = maps.sum(dim=(2, 3))                     # (B, C)

    if normalize:
        S = S / (S.sum(dim=1, keepdim=True) + 1e-8)
    if detach and not create_graph:
        S = S.detach()
    if was_training:
        model.train()
    return S


@torch.no_grad()
def predicted_logits(model, x):
    model.eval()
    return model(x)


def saliency_roi_fraction(S, roi_idx):
    """Eq. (5): ROI_frac(S) = sum_{c in R} S(c) / sum_c S(c).  S normalized."""
    roi = torch.as_tensor(roi_idx, dtype=torch.long, device=S.device)
    if S.dim() == 1:
        return S[roi].sum() / (S.sum() + 1e-8)
    return S[:, roi].sum(dim=1) / (S.sum(dim=1) + 1e-8)


def freeze_saliency(model, loader, device, roi_idx=None, **kw):
    """Pre-compute (and detach) saliency maps for a whole dataset split."""
    model.eval()
    out_S, out_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        S = channel_saliency(model, xb, class_idx=yb.to(device), **kw)
        out_S.append(S.detach().cpu())
        out_y.append(yb)
    return torch.cat(out_S), torch.cat(out_y)


def zero_filter_hook(module, inp, out, filter_idx):
    """Forward hook replacing `out` by a copy with filter `filter_idx` zeroed
    (used by NAP to evaluate SDS without touching model weights)."""
    o = out.clone()
    o[:, filter_idx] = 0.0
    return o
