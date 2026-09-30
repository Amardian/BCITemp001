"""Publication-style figures generated from live pipeline results.

Figure numbering follows the manuscript: Fig 2 (ablation), Fig 3
(accuracy-CSI Pareto), Fig 5 (accuracy-size frontier incl. baselines),
Fig 6 (causal dose-response), Fig 7a (CSI progression), Fig 7b
(coverage-accuracy-abstention trade-off), plus a Grad-CAM channel
saliency comparison (explainability preservation).

Labels are kept in English to match the manuscript figures exactly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"
ACC = "#9467bd"


def _save(fig, path):
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def fig_ablation(names, acc, size, csi, acc_drop, path, threshold=0.85):
    """Figure 2: (a) accuracy, (b) size, (c) CSI per configuration."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    x = np.arange(len(names))
    colors = [RED if d < -1.0 else BLUE for d in acc_drop]
    axes[0].bar(x, acc, color=colors)
    axes[0].set_xticks(x, names, rotation=60, ha="right")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("(a) Accuracy")
    axes[0].set_ylim(min(acc) - 2, max(acc) + 1.5)
    for i, a in enumerate(acc):
        axes[0].text(i, a + 0.2, f"{a:.1f}", ha="center", fontsize=7)
    axes[1].bar(x, size, color=[BLUE] * len(names))
    axes[1].set_xticks(x, names, rotation=60, ha="right")
    axes[1].set_ylabel("Model size (MB)")
    axes[1].set_title("(b) Size")
    for i, s in enumerate(size):
        axes[1].text(i, s + max(size) * 0.02, f"{s:.2f}", ha="center", fontsize=7)
    axes[2].bar(x, csi, color=[GREEN if c >= threshold else RED for c in csi])
    axes[2].axhline(threshold, color=RED, ls="--", lw=1)
    axes[2].text(len(names) - 0.4, threshold + 0.01, "clinical 0.85",
                 color=RED, fontsize=7, ha="right")
    axes[2].set_xticks(x, names, rotation=60, ha="right")
    axes[2].set_ylabel("CSI")
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("(c) Compression Stability Index")
    for i, c in enumerate(csi):
        axes[2].text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=7)
    return _save(fig, path)


def fig_pareto(points, path, threshold=0.85):
    """Figure 3: accuracy-CSI Pareto frontier (orderings of Table 7)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    for name, acc, csi, is_proposed in points:
        ax.scatter(csi, acc, s=90 if is_proposed else 45,
                   color=BLUE if is_proposed else GRAY,
                   marker="*" if is_proposed else "o", zorder=3)
        ax.annotate(name, (csi, acc), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    ax.axvline(threshold, color=RED, ls="--", lw=1)
    ax.text(threshold + 0.004, ax.get_ylim()[0] + 0.3, "CSI 0.85",
            color=RED, fontsize=7, rotation=90, va="bottom")
    ax.set_xlabel("CSI (saliency stability)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy-CSI Pareto frontier (orderings)")
    return _save(fig, path)


def fig_acc_size(points, path):
    """Figure 5: accuracy vs model size (all variants + baselines).

    Only the key configurations are annotated; the dense cluster of
    intermediate variants (identical size, ~equal accuracy) is listed in
    Table 6 instead, to keep the labels readable.
    """
    key_labels = {"Full Model (FP32)", "NAC (full)", "L1-only prune",
                  "EEGNet", "ShallowConvNet", "Lottery ticket",
                  "NAP only"}
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    offsets = [(8, 6), (-80, -12), (8, 2), (8, -8), (4, 8), (4, -6), (8, 0)]
    ki = 0
    for name, acc, size, kind in points:
        style = dict(main=dict(color=BLUE, marker="*", s=150),
                     baseline=dict(color=GRAY, marker="o", s=40),
                     light=dict(color=GREEN, marker="s", s=40))[kind]
        ax.scatter(size, acc, **style, zorder=3)
        if name in key_labels:
            ax.annotate(name, (size, acc), textcoords="offset points",
                        xytext=offsets[ki % len(offsets)], fontsize=7)
            ki += 1
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    ax.set_title("Accuracy-size frontier")
    ax.legend(handles=[
        plt.Line2D([], [], color=BLUE, marker="*", ls="", markersize=10,
                   label="NAC pipeline"),
        plt.Line2D([], [], color=GRAY, marker="o", ls="", markersize=6,
                   label="compression baselines"),
        plt.Line2D([], [], color=GREEN, marker="s", ls="", markersize=6,
                   label="lightweight architectures"),
    ], loc="lower right", frameon=False)
    return _save(fig, path)


def fig_dose(drops, base_acc, path, kmax=10):
    """Figure 6: causal channel-ablation dose-response."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    ks = np.arange(1, len(drops["top"]) + 1)
    ax.plot(ks, 100 * np.array(drops["top"]), "o-", color=RED,
            label="top-k saliency channels (causal)")
    ax.plot(ks, 100 * np.array(drops["random"]), "s--", color=GRAY,
            label="random-k channels")
    ax.plot(ks, 100 * np.array(drops["bottom"]), "^:", color=GREEN,
            label="bottom-k saliency channels")
    ax.set_xlabel("k (channels ablated)")
    ax.set_ylabel("Accuracy drop (pp)")
    ax.set_title("Dose-response of causal channel ablation")
    ax.legend(frameon=False, loc="upper left")
    ax.axhline(0, color="k", lw=0.5)
    return _save(fig, path)


def fig_csi_progression(stages, csi, ci_lo, ci_hi, path, threshold=0.85):
    """Figure 7a: CSI progression through the NAC pipeline."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    x = np.arange(len(stages))
    ax.errorbar(x, csi, yerr=[np.array(csi) - np.array(ci_lo),
                              np.array(ci_hi) - np.array(csi)],
                fmt="o-", color=BLUE, capsize=4, lw=1.5)
    ax.axhline(threshold, color=RED, ls="--", lw=1)
    ax.text(x[-1] + 0.05, threshold, "0.85", color=RED, fontsize=7,
            va="center")
    ax.set_xticks(x, stages, rotation=20, ha="right")
    ax.set_ylabel("CSI (95% BCa CI)")
    ax.set_ylim(0, 1.08)
    ax.set_title("CSI progression through the NAC pipeline")
    return _save(fig, path)


def fig_tradeoff(sweep, path):
    """Figure 7b: coverage / committed accuracy / abstention vs tau_commit."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    tc = [s["tau_commit"] for s in sweep]
    ax.plot(tc, [s["coverage"] for s in sweep], "o-", color=BLUE,
            label="Coverage (%)")
    ax.plot(tc, [s["committed_accuracy"] for s in sweep], "s-",
            color=GREEN, label="Committed-trial accuracy (%)")
    ax.plot(tc, [s["abstention"] for s in sweep], "^--", color=RED,
            label="Abstention (%)")
    ax.axvline(0.7, color=GRAY, ls=":", lw=1)
    ax.text(0.705, ax.get_ylim()[0] + 3, r"$\tau_{commit}=0.7$", fontsize=7)
    ax.set_xlabel(r"Commit threshold $\tau_{commit}$")
    ax.set_ylabel("Percent")
    ax.set_title("Coverage-accuracy-abstention trade-off")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, path)


def fig_saliency(channels, S_full, S_nac, roi_idx, path, top=20):
    """Grad-CAM channel saliency: full vs compressed model (top channels)."""
    order = np.argsort(-S_full)
    top_idx = order[:top]
    others = order[top:]
    S_full_rest = float(S_full[others].sum())
    S_nac_rest = float(S_nac[others].sum())
    names = [channels[i] for i in top_idx] + ["rest"]
    vf = np.append(S_full[top_idx], S_full_rest)
    vn = np.append(S_nac[top_idx], S_nac_rest)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    w = 0.4
    colors = [ACC if i in set(roi_idx.tolist()) else BLUE
              for i in top_idx] + [GRAY]
    ax.bar(x - w / 2, vf, width=w, color=colors, alpha=0.85,
           label="Full FP32 (teacher)")
    ax.bar(x + w / 2, vn, width=w, color=ORANGE, alpha=0.85,
           label="NAC (compressed)")
    ax.set_xticks(x, names, rotation=60, ha="right")
    ax.set_ylabel("Mean Grad-CAM saliency (L1-normalized)")
    ax.set_title("Channel saliency: full vs compressed model")
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, path)
