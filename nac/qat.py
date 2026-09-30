"""Stage 2: INT8 Quantization-Aware Training (QAT) - Section 4.3.2, Alg. 2.

Eq. (8):  Q(x) = round(x/s)*s,  s = (x_max-x_min)/(2^b-1),  b=8
Eq. (9):  straight-through estimator: dQ/dx = 1 inside [x_min, x_max],
          0 outside.

Per-channel quantization for weights (conv + linear), per-tensor for
activations (paper Section 4.3.2). Calibration on cfg.qat_calib samples,
then fine-tuning with STE.

`quantize_model(model, calib_X, mode)` returns a fake-quantized copy:
  mode='qat'  : observer calibration + STE fake-quant in forward,
                differentiable -> used for QAT fine-tuning and SPKD
  mode='ptq'  : post-training quantization (no fine-tuning baseline)
`int8_size_mb` accounts 1 byte/param.

Note (documented deviation): the demo executes the GRU in FP32 with
INT8-storage accounting; full INT8 GRU kernels require TensorRT/ONNX
engines as in the paper's Jetson deployment.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class _STE(torch.autograd.Function):
    """Straight-through estimator with range gating (Eq. 9)."""

    @staticmethod
    def forward(ctx, x, xmin, xmax):
        ctx.save_for_backward(x, xmin, xmax)
        qmin, qmax = -127.0, 127.0
        scale = (xmax - xmin) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-12)
        zp = torch.round(qmin - xmin / scale)
        q = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
        return (q - zp) * scale

    @staticmethod
    def backward(ctx, g):
        x, xmin, xmax = ctx.saved_tensors
        inside = ((x >= xmin) & (x <= xmax)).to(x.dtype)
        return g * inside, None, None


def _fake_quant(x, xmin, xmax):
    return _STE.apply(x, xmin, xmax)


class FakeQuantConv2d(nn.Conv2d):
    """Conv2d with per-channel weight + per-tensor activation fake-quant."""

    def __init__(self, conv: nn.Conv2d, per_channel_weight=True):
        kw = dict(kernel_size=conv.kernel_size, stride=conv.stride,
                  padding=conv.padding, dilation=conv.dilation,
                  groups=conv.groups, bias=conv.bias is not None)
        super().__init__(conv.in_channels, conv.out_channels, **kw)
        with torch.no_grad():
            self.weight.copy_(conv.weight.detach())
            if conv.bias is not None:
                self.bias.copy_(conv.bias.detach())
        self.per_channel_weight = per_channel_weight
        self.observer = True          # calibrating
        self.quant_enabled = False    # fake-quant active
        self.register_buffer("a_min", torch.zeros(1))
        self.register_buffer("a_max", torch.zeros(1))
        self.register_buffer("w_min", torch.zeros(self.out_channels))
        self.register_buffer("w_max", torch.zeros(self.out_channels))

    def _observe(self, x):
        amin = x.detach().min().item()
        amax = x.detach().max().item()
        if self.a_min.item() == 0.0 and self.a_max.item() == 0.0:
            self.a_min.fill_(amin)
            self.a_max.fill_(amax)
        else:
            lo = 0.9 * self.a_min.item() + 0.1 * amin
            hi = 0.9 * self.a_max.item() + 0.1 * amax
            self.a_min.fill_(lo)
            self.a_max.fill_(hi)

    def forward(self, x):
        if self.observer:
            self._observe(x)
        if not self.quant_enabled:
            return super().forward(x)
        if self.per_channel_weight:
            wshape = (self.out_channels,) + (1,) * (self.weight.dim() - 1)
            w = _fake_quant(self.weight, self.w_min.view(wshape),
                            self.w_max.view(wshape))
        else:
            w = _fake_quant(self.weight, self.w_min.min(),
                            self.w_max.max())
        xa = _fake_quant(x, self.a_min, self.a_max)
        return F.conv2d(xa, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class FakeQuantLinear(nn.Linear):
    """Linear with per-channel weight + per-tensor input fake-quant."""

    def __init__(self, lin: nn.Linear):
        super().__init__(lin.in_features, lin.out_features,
                         bias=lin.bias is not None)
        with torch.no_grad():
            self.weight.copy_(lin.weight.detach())
            if lin.bias is not None:
                self.bias.copy_(lin.bias.detach())
        self.observer = True
        self.quant_enabled = False
        self.register_buffer("a_min", torch.zeros(1))
        self.register_buffer("a_max", torch.zeros(1))
        self.register_buffer("w_min", torch.zeros(self.out_features))
        self.register_buffer("w_max", torch.zeros(self.out_features))

    def forward(self, x):
        if self.observer:
            amin, amax = x.detach().min().item(), x.detach().max().item()
            if self.a_min.item() == 0.0 and self.a_max.item() == 0.0:
                self.a_min.fill_(amin)
                self.a_max.fill_(amax)
            else:
                self.a_min.fill_(0.9 * self.a_min.item() + 0.1 * amin)
                self.a_max.fill_(0.9 * self.a_max.item() + 0.1 * amax)
        if not self.quant_enabled:
            return super().forward(x)
        wshape = (self.out_features, 1)
        w = _fake_quant(self.weight, self.w_min.view(wshape),
                        self.w_max.view(wshape))
        xa = _fake_quant(x, self.a_min, self.a_max)
        return F.linear(xa, w, self.bias)


def quantize_model(model, calib_X, mode="qat", device="cpu", logger=print):
    """Return a fake-quantized deep copy of `model`.

    mode='qat'/'ptq': run calibration forward passes on calib_X (weight
    ranges from the tensors themselves; activation ranges via observers),
    then enable fake-quant. The copy is differentiable through the STE,
    so QAT fine-tuning and SPKD work on it directly.
    """
    q = copy.deepcopy(model).to(device)

    def convert_module(mod, cls, **kw):
        new = cls(mod, **kw)
        # initialize weight ranges
        w = mod.weight.detach()
        new.w_min.copy_(w.reshape(w.shape[0], -1).min(dim=1).values)
        new.w_max.copy_(w.reshape(w.shape[0], -1).max(dim=1).values)
        return new

    # replace convs inside ConvBlocks
    for i, blk in enumerate(q.tower):
        blk.conv = convert_module(blk.conv, FakeQuantConv2d,
                                  per_channel_weight=True)
    q.embedding = convert_module(q.embedding, FakeQuantLinear)
    if getattr(q, "spatial", None) is not None:
        q.spatial = convert_module(q.spatial, FakeQuantLinear)
    q.head = convert_module(q.head, FakeQuantLinear)

    # calibration pass (activations)
    q.eval()
    with torch.no_grad():
        for s in range(0, len(calib_X), 64):
            q(calib_X[s:s + 64].to(device))
    # freeze observers, enable fake-quant
    for m in q.modules():
        if hasattr(m, "observer"):
            m.observer = False
            m.quant_enabled = True
    if mode == "ptq":
        logger("  [QAT] PTQ mode: quantization applied without fine-tuning")
    return q


def int8_size_mb(model) -> float:
    """INT8 model size: 1 byte per parameter + scale/zero-point bytes."""
    n = sum(p.numel() for p in model.parameters())
    scale_bytes = 0
    for m in model.modules():
        if hasattr(m, "quant_enabled") and getattr(m, "quant_enabled", False):
            scale_bytes += 8 * 4  # a/w min-max buffers
    return (n * 1 + scale_bytes) / 1024 ** 2


def fake_quant_state(model):
    return {k: v for k, v in model.state_dict().items()
            if "w_min" in k or "w_max" in k}
