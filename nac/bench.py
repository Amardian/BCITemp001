"""Edge-deployment benchmarking (Section 4.7).

Measured on the host CPU of this run (paper reports Jetson Nano via
TensorRT INT8 and Raspberry Pi 4 via ONNX-Runtime; those require the
physical devices - `export_onnx` is provided for that path).

Metrics: latency mean/std/p95/p99 over `latency_runs` runs after warm-up,
throughput, simulated 30-minute sustained load (latency drift), and the
online sliding-window protocol (500 ms window, 100 ms step, majority
vote over 5 windows).
"""

from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def latency_benchmark(model, X, device, warmup=100, runs=900):
    model.eval().to(device)
    for _ in range(warmup):
        model(X[:1].to(device))
    lat = []
    for i in range(runs):
        xi = X[i % len(X):i % len(X) + 1].to(device)
        t0 = time.perf_counter()
        model(xi)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat)
    return dict(
        mean_ms=float(lat.mean()), std_ms=float(lat.std()),
        p95_ms=float(np.percentile(lat, 95)), p99_ms=float(np.percentile(lat, 99)),
        throughput_per_s=float(1000.0 / lat.mean()),
        n_runs=runs, warmup=warmup,
    )


@torch.no_grad()
def sustained_drift(model, X, device, runs=1000):
    """Simulated 30-min sustained inference: drift = last-100 mean vs
    first-100 mean latency (Table 9 'latency drift')."""
    model.eval().to(device)
    lat = []
    for i in range(runs):
        xi = X[i % len(X):i % len(X) + 1].to(device)
        t0 = time.perf_counter()
        model(xi)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat)
    first, last = lat[:100].mean(), lat[-100:].mean()
    return dict(runs=runs, first100_ms=float(first), last100_ms=float(last),
                drift_pct=float(100 * (last - first) / first))


@torch.no_grad()
def sliding_window_online(model, X, y, device, window_len=8, step=2,
                          n_windows=5):
    """Online protocol: split each trial's time axis into n_windows
    overlapping windows (window_len bins, step bins), predict per window,
    majority vote (Table 10)."""
    model.eval().to(device)
    T = X.shape[-1]
    starts = [i * step for i in range(n_windows)]
    votes = []
    for i in range(len(y)):
        preds = []
        for st in starts:
            if st + window_len > T:
                st = max(0, T - window_len)
            xi = X[i:i + 1, ..., st:st + window_len].to(device)
            if xi.shape[-1] < window_len:
                continue
            preds.append(int(model(xi).argmax(1).item()))
        if preds:
            vals, counts = np.unique(preds, return_counts=True)
            votes.append(vals[counts.argmax()])
        else:
            votes.append(-1)
    votes = np.array(votes)
    return dict(online_accuracy=float((votes == y.numpy()).mean()),
                n_windows=n_windows, window_len=window_len, step=step,
                starts=starts)


def export_onnx(model, X_sample, path):
    """Export the model to ONNX (opset 13) for TensorRT / ONNX-Runtime
    deployment on Jetson Nano / Raspberry Pi 4 (Section 4.7)."""
    model = model.cpu().eval()
    torch.onnx.export(model, X_sample[:1], path, opset_version=13,
                      input_names=["cwt_features"],
                      output_names=["logits"], dynamo=False)
    return path
