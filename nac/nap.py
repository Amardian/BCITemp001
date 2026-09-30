"""Stage 1: Neurophysiology-Aware Pruning (NAP) - Section 4.3.1, Alg. 1.

SDS (Eq. 6)  : |ROI_frac(S_full) - ROI_frac(S_{l,i})|  for filter F_{l,i}
               zeroed, where S is the Grad-CAM channel saliency.
Importance   : lambda*SDS + (1-lambda)*NormL1  (Eq. 7, lambda=0.6)
Iterative    : K iterations, prune s/K of the ORIGINAL filters each time,
               fine-tune after each iteration (Alg. 1).

Pruning is *structured* (filters physically removed): every conv layer is
rebuilt with the kept filters and the next layer's in-channels are sliced,
so the size/speed reductions are real, not masks.

Baselines: L1-only pruning (criterion = NormL1) and the lottery-ticket
hypothesis (LTH: prune at init by L1, retrain from init).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .gradcam import channel_saliency, saliency_roi_fraction, zero_filter_hook


def norm_l1(conv: nn.Conv2d) -> np.ndarray:
    """Per-filter L1 norm normalized by the layer max (for scale-compatible
    mixing with SDS in [0, 1])."""
    w = conv.weight.detach().abs().sum(dim=(1, 2, 3)).cpu().numpy()
    return w / (w.max() + 1e-12)


@torch.enable_grad()
def sds_scores(model, X_sds, y_sds, roi_idx):
    """Exact SDS for every filter of every prunable conv layer.

    For each filter: forward the SDS subset with that filter zeroed
    (via a forward hook), compute Grad-CAM saliency, measure the change
    in ROI_frac vs. the full model (Eq. 6).
    Returns dict layer_name -> (n_filters,) array of SDS.
    """
    model.eval()
    # baseline saliency of the unmodified model
    S_full = channel_saliency(model, X_sds, class_idx=y_sds, normalize=True)
    roi_full = saliency_roi_fraction(S_full, roi_idx).detach()

    layers = list(model.tower)
    sds = {}
    for li, conv in enumerate(layers):
        n = conv.conv.out_channels
        scores = torch.zeros(n)
        for i in range(n):
            hook = conv.conv.register_forward_hook(
                lambda m, inp, out, idx=i: zero_filter_hook(m, inp, out, idx))
            try:
                S_i = channel_saliency(model, X_sds, class_idx=y_sds,
                                       normalize=True)
            finally:
                hook.remove()
            roi_i = saliency_roi_fraction(S_i, roi_idx).detach()
            scores[i] = (roi_full - roi_i).abs().mean()
        sds[f"tower.{li}"] = scores.numpy()
    return sds


def importance_scores(model, X_sds, y_sds, roi_idx, lam=0.6, criterion="nap"):
    """Importance = lambda*SDS + (1-lambda)*NormL1  (Eq. 7).

    criterion='nap' -> Eq. 7;  criterion='l1' -> NormL1 only (baseline).
    Returns (dict layer -> importance, dict layer -> sds, dict layer -> l1).
    """
    sds = sds_scores(model, X_sds, y_sds, roi_idx)
    l1 = {f"tower.{i}": norm_l1(blk.conv) for i, blk in enumerate(model.tower)}
    if criterion == "l1":
        imp = {k: v for k, v in l1.items()}
    else:
        imp = {k: lam * sds[k] + (1 - lam) * l1[k] for k in sds}
    return imp, sds, l1


def select_prune_mask(model, imp, n_prune):
    """Pick the n_prune globally lowest-importance filters.

    Per-iteration quota (Alg. 1 line 7): prune the lowest s/K fraction of
    the ORIGINAL filter count, spread across layers proportional to size.
    Returns keep_masks: dict layer_name -> boolean array (True = keep).
    """
    total = sum(len(v) for v in imp.values())
    quota = {k: max(1, int(round(n_prune * len(v) / total))) for k, v in imp.items()}
    # trim to exactly n_prune if rounding overshot
    excess = sum(quota.values()) - n_prune
    keys_sorted = sorted(quota, key=lambda k: len(imp[k]))
    ki = 0
    while excess > 0 and any(quota[k] > 1 for k in keys_sorted):
        k = keys_sorted[ki % len(keys_sorted)]
        if quota[k] > 1:
            quota[k] -= 1
            excess -= 1
        ki += 1
    keep = {}
    for k, v in imp.items():
        order = np.argsort(v)             # ascending importance
        drop = set(order[:quota[k]].tolist())
        keep[k] = np.array([i not in drop for i in range(len(v))])
    return keep


def apply_structured_prune(model, keep_masks):
    """Physically remove filters (rebuild conv layers, slice next in-channels).

    keep_masks keys are 'tower.<i>' boolean arrays over that block's conv
    filters. The last block also slices the channel-embedding in-features.
    Returns a NEW model (deep copy) with the same head/GRU weights.
    """
    pruned = copy.deepcopy(model)
    n_blocks = len(pruned.tower)
    for bi in range(n_blocks):
        key = f"tower.{bi}"
        if key not in keep_masks:
            continue
        keep = torch.as_tensor(keep_masks[key], dtype=torch.bool)
        if bool(keep.all()):
            continue
        conv = pruned.tower[bi].conv
        new_conv = nn.Conv2d(
            conv.in_channels, int(keep.sum()),
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight[keep])
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias[keep])
        new_conv = new_conv.to(conv.weight.device)
        pruned.tower[bi].conv = new_conv
        # slice the consumer
        if bi + 1 < n_blocks:
            nxt = pruned.tower[bi + 1].conv
            new_next = nn.Conv2d(
                int(keep.sum()), nxt.out_channels,
                kernel_size=nxt.kernel_size, stride=nxt.stride,
                padding=nxt.padding, bias=nxt.bias is not None)
            with torch.no_grad():
                new_next.weight.copy_(nxt.weight[:, keep])
                if nxt.bias is not None:
                    new_next.bias.copy_(nxt.bias)
            new_next = new_next.to(nxt.weight.device)
            pruned.tower[bi + 1].conv = new_next
        else:
            emb = pruned.embedding
            new_emb = nn.Linear(int(keep.sum()), emb.out_features,
                                bias=emb.bias is not None)
            with torch.no_grad():
                new_emb.weight.copy_(emb.weight[:, keep])
                if emb.bias is not None:
                    new_emb.bias.copy_(emb.bias)
            new_emb = new_emb.to(emb.weight.device)
            pruned.embedding = new_emb
    return pruned


def nap(model, train_loader, sds_X, sds_y, roi_idx, cfg, device,
        criterion="nap", logger=print):
    """Algorithm 1: iterative saliency-preserving structured pruning.

    K iterations; each prunes (s/K * original filters) with the chosen
    criterion, then fine-tunes for cfg.nap_ft_epochs at LR 1e-4.
    Returns (pruned_model, info dict).
    """
    current = copy.deepcopy(model)
    total_filters = sum(b.conv.out_channels for b in current.tower)
    n_total_prune = int(round(cfg.nap_sparsity * total_filters))
    per_iter = max(1, n_total_prune // cfg.nap_iterations)
    info = dict(criterion=criterion, total_filters=total_filters,
                per_iteration=per_iter, history=[])
    for it in range(cfg.nap_iterations):
        imp, sds, l1 = importance_scores(
            current, sds_X.to(device), sds_y.to(device), roi_idx,
            lam=cfg.nap_lambda, criterion=criterion)
        keep = select_prune_mask(current, imp, per_iter)
        current = apply_structured_prune(current, keep)
        kept = sum(b.conv.out_channels for b in current.tower)
        logger(f"  [NAP {criterion}] iter {it + 1}/{cfg.nap_iterations}: "
               f"kept {kept}/{total_filters} filters "
               f"({100 * kept / total_filters:.1f}%)")
        info["history"].append(dict(iteration=it + 1, kept=kept))
        if cfg.nap_ft_epochs > 0:
            from .train import finetune  # local import to avoid cycle
            finetune(current, train_loader, device,
                     epochs=cfg.nap_ft_epochs, lr=cfg.nap_ft_lr,
                     logger=lambda m: None)
    return current, info


def l1_prune(model, train_loader, cfg, device, logger=print):
    """L1-only structured pruning baseline (same schedule, NormL1 criterion).

    Uses magnitude pruning without SDS (Section 6.1, 'Pruned (40%, L1-only)'
    row of Table 6).
    """
    current = copy.deepcopy(model)
    total_filters = sum(b.conv.out_channels for b in current.tower)
    n_total_prune = int(round(cfg.nap_sparsity * total_filters))
    per_iter = max(1, n_total_prune // cfg.nap_iterations)
    for it in range(cfg.nap_iterations):
        l1 = {f"tower.{i}": norm_l1(blk.conv)
              for i, blk in enumerate(current.tower)}
        keep = select_prune_mask(current, l1, per_iter)
        current = apply_structured_prune(current, keep)
        logger(f"  [L1] iter {it + 1}: kept "
               f"{sum(b.conv.out_channels for b in current.tower)}/{total_filters}")
        if cfg.nap_ft_epochs > 0:
            from .train import finetune
            finetune(current, train_loader, device,
                     epochs=cfg.l1_ft_epochs, lr=cfg.nap_ft_lr,
                     logger=lambda m: None)
    return current


def lottery_ticket(model_init, train_loader, cfg, device, logger=print):
    """Lottery-ticket baseline: prune at initialization by L1, retrain from
    the original initialization weights."""
    lth = copy.deepcopy(model_init)
    total_filters = sum(b.conv.out_channels for b in lth.tower)
    n_prune = int(round(cfg.nap_sparsity * total_filters))
    l1 = {f"tower.{i}": norm_l1(blk.conv) for i, blk in enumerate(lth.tower)}
    keep = select_prune_mask(lth, l1, n_prune)
    lth = apply_structured_prune(lth, keep)
    from .train import finetune
    finetune(lth, train_loader, device, epochs=cfg.lth_epochs, lr=cfg.lr,
             logger=lambda m: None)
    return lth
