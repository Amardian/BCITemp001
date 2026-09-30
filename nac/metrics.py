"""Metrics and statistics.

CSI / CSI-S / SPR (Definitions 4.2-4.3, Eqs. 17-19):
  CSI_trial  = 1 - ||S_full - S_comp||_1 / ||S_full||_1     (both L1-norm)
  CSI-S      = (Spearman(S_full, S_comp) + 1) / 2
  SPR        = ROI_frac(S_comp) / ROI_frac(S_full)
  Clinical threshold CSI >= 0.85.

Statistics (Section 5.2): BCa bootstrap CIs, Wilcoxon signed-rank,
Cohen's d with bootstrap CI, permutation tests, non-inferiority with
pre-registered margin delta = 1%.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- CSI family
def csi_trial(S_full, S_comp):
    """Eq. 17. Inputs (C,) or (N, C) L1-normalized saliency."""
    l1 = np.abs(S_full - S_comp).sum(axis=-1)
    denom = np.abs(S_full).sum(axis=-1) + 1e-12
    return 1.0 - l1 / denom


def cisi_s_trial(S_full, S_comp):
    """Eq. 18 (rank-based, outlier-robust)."""
    S_full = np.atleast_2d(S_full)
    S_comp = np.atleast_2d(S_comp)
    rhos = []
    for i in range(S_full.shape[0]):
        rho = stats.spearmanr(S_full[i], S_comp[i]).statistic
        rhos.append(0.0 if np.isnan(rho) else rho)
    return np.array(rhos)


def spr_trial(S_full, S_comp, roi_idx):
    """Eq. 19: ratio of ROI-concentrated saliency comp / full."""
    roi = np.asarray(roi_idx, dtype=int)
    roi_f = S_full[..., roi].sum(axis=-1) / (np.abs(S_full).sum(axis=-1) + 1e-12)
    roi_c = S_comp[..., roi].sum(axis=-1) / (np.abs(S_comp).sum(axis=-1) + 1e-12)
    return roi_c / (roi_f + 1e-12)


def compression_stability(S_full, S_comp, roi_idx):
    """Per-trial CSI, CSI-S, SPR + summaries for a test split.

    Trials whose Grad-CAM mass is entirely zero in either model (ReLU
    kills every pre-activation - can happen for barely-trained models)
    are excluded from the summaries so the metrics stay defined;
    `n_valid` records how many trials entered the statistics.
    """
    import warnings
    S_full = np.asarray(S_full)
    S_comp = np.asarray(S_comp)
    roi = np.asarray(roi_idx, dtype=int)
    roi_f_all = S_full[..., roi].sum(axis=-1) / \
        (np.abs(S_full).sum(axis=-1) + 1e-12)
    valid = (np.abs(S_full).sum(axis=-1) > 1e-9) & \
            (np.abs(S_comp).sum(axis=-1) > 1e-9) & \
            (roi_f_all > 1e-4)      # SPR undefined without teacher ROI mass
    n_valid = int(valid.sum())
    if n_valid == 0:
        return dict(per_trial=dict(csi=np.zeros(len(valid)),
                                   csi_s=np.zeros(len(valid)),
                                   spr=np.zeros(len(valid))),
                    csi_mean=float("nan"), csi_s_mean=float("nan"),
                    spr_mean=float("nan"), csi_std=float("nan"),
                    csi_s_std=float("nan"), spr_std=float("nan"),
                    n_valid=0)
    Sf, Sc = S_full[valid], S_comp[valid]
    csi = csi_trial(Sf, Sc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csis = cisi_s_trial(Sf, Sc)
    spr = spr_trial(Sf, Sc, roi_idx)
    return dict(
        per_trial=dict(csi=csi, csi_s=csis, spr=spr),
        csi_mean=float(csi.mean()), csi_s_mean=float(csis.mean()),
        spr_mean=float(spr.mean()),
        csi_std=float(csi.std()), csi_s_std=float(csis.std()),
        spr_std=float(spr.std()), n_valid=n_valid,
    )


# ---------------------------------------------------------------- calibration
def expected_calibration_error(probs, y, n_bins=15):
    """ECE with equal-width confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


# ---------------------------------------------------------------- bootstrap
def bca_bootstrap(x, stat=np.mean, n_boot=10000, alpha=0.05, seed=0):
    """Bias-corrected and accelerated (BCa) confidence interval."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    theta = stat(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([stat(x[i]) for i in idx])
    # bias-correction z0
    z0 = stats.norm.ppf(np.mean(boot < theta) + 1e-12)
    # acceleration via jackknife
    jk = np.array([stat(np.delete(x, i)) for i in range(n)])
    jmean = jk.mean()
    denom = np.sum((jmean - jk) ** 2) ** 1.5
    a = np.sum((jmean - jk) ** 3) / (2 * denom + 1e-12) if denom > 0 else 0.0
    probs = ((z0 + stats.norm.ppf(alpha / 2)) /
             (1 - a * (z0 + stats.norm.ppf(alpha / 2))))
    lo_q = stats.norm.cdf((z0 + stats.norm.ppf(alpha / 2)) /
                          (1 - a * (z0 + stats.norm.ppf(alpha / 2)))) * 100
    hi_q = stats.norm.cdf((z0 + stats.norm.ppf(1 - alpha / 2)) /
                          (1 - a * (z0 + stats.norm.ppf(1 - alpha / 2)))) * 100
    lo_q = np.clip(lo_q, 0, 100)
    hi_q = np.clip(hi_q, 0, 100)
    ci = (np.percentile(boot, lo_q), np.percentile(boot, hi_q))
    return dict(point=float(theta), ci_low=float(ci[0]),
                ci_high=float(ci[1]), n_boot=n_boot)


def percentile_ci(x, stat=np.mean, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(stat(x)), ci_low=float(lo), ci_high=float(hi))


# ---------------------------------------------------------------- tests
def wilcoxon_signed(x, y=None, alternative="two-sided"):
    """Wilcoxon signed-rank across subjects (paired accuracies)."""
    x = np.asarray(x)
    if y is not None:
        x = x - np.asarray(y)
    try:
        res = stats.wilcoxon(x, alternative=alternative,
                             zero_method="wilcox")
        return dict(statistic=float(res.statistic), p=float(res.pvalue))
    except ValueError:  # all differences zero
        return dict(statistic=0.0, p=1.0)


def cohens_dz(x, y=None):
    """Cohen's d_z for paired samples (with BCa CI on the mean diff)."""
    if y is not None:
        d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    else:
        d = np.asarray(x, dtype=float)
    if d.std(ddof=1) == 0:
        return 0.0
    return float(d.mean() / d.std(ddof=1))


def permutation_test(x, y=None, n_perm=1000, seed=0, alternative="greater"):
    """Paired permutation test on the mean difference."""
    rng = np.random.default_rng(seed)
    if y is not None:
        d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    else:
        d = np.asarray(x, dtype=float)
    obs = d.mean()
    null = []
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(d))
        null.append((d * s).mean())
    null = np.array(null)
    if alternative == "greater":
        p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    elif alternative == "less":
        p = float((np.sum(null <= obs) + 1) / (n_perm + 1))
    else:
        p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return dict(observed=float(obs), p=p, n_perm=n_perm)


def noninferiority(diff, margin=0.01, alpha=0.025):
    """One-sided non-inferiority test: mean(diff) > -margin.

    Uses the t-based CI on the paired differences (Section 5.2, item i:
    pre-registered delta = 1%).
    """
    d = np.asarray(diff, dtype=float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    t_crit = stats.t.ppf(1 - alpha, df=max(n - 1, 1))
    ci_low = mean - t_crit * se
    return dict(mean=float(mean), ci_low=float(ci_low),
                margin=-margin, noninferior=bool(ci_low > -margin))
