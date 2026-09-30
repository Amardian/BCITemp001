"""Model architectures.

Base classifier (paper Section 4.2) - CNN-GRU hybrid:
  "four convolutional blocks (32, 64, 128, 256 filters; 3x3 kernels; ELU
   activation; 2x2 average pooling), a two-layer GRU (128 hidden units),
   and a softmax classification head."

Implementation mapping: the CWT map of every EEG channel is passed
through the shared conv tower; each channel is then embedded, and the
GRU runs over the channel sequence. This keeps one feature map A^k per
EEG channel, so Grad-CAM (Eq. 14-15) yields a saliency value per channel
S(c) - exactly the quantity used by ROI_frac (Eq. 5), NAP's SDS (Eq. 6),
SPKD's saliency-alignment loss (Eq. 12) and CSI/SPR (Eqs. 17-19).

Lightweight baselines (paper Section 5.3, category 3): EEGNet,
ShallowConvNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3 -> ELU -> AvgPool2x2 (paper Section 4.2).

    ceil_mode keeps spatial dims >= 1 so short sliding windows
    (Section 4.7 online protocol) still produce valid feature maps.
    """

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.pool(F.elu(self.conv(x)))


class CNNGRU(nn.Module):
    """Channel-sequence CNN-GRU (base architecture, Section 4.2).

    The CWT map of every EEG channel goes through the shared conv tower;
    the per-channel embeddings feed (a) the two-layer GRU running over the
    channel sequence and (b) a spatial readout (flattened channel x filter
    features -> Linear), whose role is the learned spatial filter that
    mixes channels - the EEG analogue of a CSP/conv1 spatial projection.
    Both branches are concatenated into the classification head.

    Keeping one feature-map stack per EEG channel makes Grad-CAM
    (Eq. 14-15) yield a saliency value per channel S(c), which is exactly
    the quantity used by ROI_frac (Eq. 5), NAP's SDS (Eq. 6), SPKD's
    L_SAL (Eq. 12) and CSI/SPR (Eqs. 17-19).
    """

    def __init__(self, n_channels, n_classes, n_freqs=16, n_times=16,
                 filters=(32, 64, 128, 256), gru_hidden=128, gru_layers=2,
                 embedding_dim=160, spatial_dim=160):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        blocks = []
        cin = 1
        for f in filters:
            blocks.append(ConvBlock(cin, f))
            cin = f
        self.tower = nn.ModuleList(blocks)
        self.embedding = nn.Linear(filters[-1], embedding_dim)
        self.gru = nn.GRU(embedding_dim, gru_hidden, num_layers=gru_layers,
                          batch_first=True)
        self.spatial = nn.Linear(n_channels * embedding_dim, spatial_dim)
        self.head = nn.Linear(gru_hidden + spatial_dim, n_classes)

    def forward_features(self, x):
        """x: (B, C, F, T) -> per-channel feature maps (B, C, K, f, t)."""
        B, C = x.shape[:2]
        z = x.reshape(B * C, 1, *x.shape[2:])
        for blk in self.tower:
            z = blk(z)
        return z.reshape(B, C, *z.shape[1:])     # (B, C, K, f, t)

    def forward_head(self, z):
        """z: (B, C, K, f, t) -> logits (shares the graph of z)."""
        z2 = z.mean(dim=(3, 4))                     # GAP -> (B, C, K)
        emb = F.elu(self.embedding(z2))             # (B, C, D)
        out, _ = self.gru(emb)
        h_gru = out[:, -1]                          # (B, gru_hidden)
        sp = self.spatial(emb.flatten(1))           # spatial readout (B, S)
        return self.head(torch.cat([h_gru, sp], dim=1))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model, mode: str = "fp32") -> float:
    """Size in MB. 'fp32' = 4 B/param; 'int8' = 1 B/param (+ scales).

    Quantized scales/zero-points add O(few hundred) bytes and are counted
    from the module's quant state when present.
    """
    n = count_parameters(model)
    scale_bytes = 0
    for m in model.modules():
        if hasattr(m, "quant_enabled") and getattr(m, "quant_enabled", False):
            scale_bytes += 2 * 4  # min/max stored per quantized tensor
    if mode == "fp32":
        return (n * 4 + scale_bytes) / 1024 ** 2
    return (n * 1 + scale_bytes) / 1024 ** 2


# --------------------------------------------------------------------------
# Lightweight baselines (Section 5.3)
# --------------------------------------------------------------------------
class EEGNet(nn.Module):
    """EEGNet (Lawhern et al. 2018) - compact baseline (~2-3K params).

    Input X (B, C, F, T) is unfolded to (B, 1, C, F*T): the CWT channels
    play the role of EEG electrodes and (freq x time) the temporal axis.
    """

    def __init__(self, n_channels, n_classes, n_times=256, f1=8, f2=16):
        super().__init__()
        self.firstconv = nn.Conv2d(1, f1, kernel_size=(1, 25), padding=(0, 12),
                                   bias=False)
        self.depthwise = nn.Conv2d(f1, f1, kernel_size=(n_channels, 1),
                                   groups=f1, bias=False)
        self.sepdepth = nn.Conv2d(f1, f1, kernel_size=(1, 15), padding=(0, 7),
                                  groups=f1, bias=False)
        self.seppoint = nn.Conv2d(f1, f2, kernel_size=(1, 1), bias=False)
        t = n_times
        t = (t + 1) // 1                       # temporal conv keeps length
        t = t // 4                              # avgpool 4
        t = t // 8                              # avgpool 8
        self.classify = nn.Linear(f2 * max(1, t), n_classes)

    def forward(self, x):
        B = x.shape[0]
        z = x.reshape(B, 1, x.shape[1], -1)     # (B, 1, C, F*T)
        z = F.elu(self.firstconv(z))
        z = F.elu(self.depthwise(z))
        z = F.avg_pool2d(z, (1, 4))
        z = F.elu(self.sepdepth(z))
        z = F.avg_pool2d(z, (1, 8))
        z = F.elu(self.seppoint(z))
        z = z.flatten(1)
        return self.classify(z)


class ShallowConvNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al. 2017)."""

    def __init__(self, n_channels, n_classes, n_times=256, n_filters=40):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters, kernel_size=(1, 25),
                                   padding=(0, 12))
        self.conv_spat = nn.Conv2d(n_filters, n_filters,
                                   kernel_size=(n_channels, 1),
                                   groups=n_filters, bias=False)
        t = n_times // 10
        self.classify = nn.Linear(n_filters * max(1, t), n_classes)

    def forward(self, x):
        B = x.shape[0]
        z = x.reshape(B, 1, x.shape[1], -1)      # (B, 1, C, F*T)
        z = self.conv_time(z)
        z = self.conv_spat(z)                     # (B, nf, 1, T)
        z = z.squeeze(2)
        z = F.selu(z)
        z = F.avg_pool1d(z, kernel_size=10, stride=10)
        z = z.flatten(1)
        return self.classify(z)
