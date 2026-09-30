"""Datasets, montage/ROI definitions and data simulation.

Paper (Section 5.1):
  BCI Competition IV-2a : transfer source, 9 subjects, 22 ch, 250 Hz,
                          4-class limb MI, 576 trials/subject.
  KaraOne               : primary SMI target, 12 subjects, 64 ch,
                          1000 Hz, 11 classes (7 phonemes + 4 words).
  Preprocessing: 0.5-45 Hz bidirectional Butterworth, ICA artifact
  rejection, Morlet CWT (w0=6), z-score per channel.

Because the real datasets are not redistributable, this module provides
(a) loaders for the real data (`load_karaone_mat`, `load_bci2a_gdf`) and
(b) a neurophysiology-plausible simulator used by default, which plants
class-discriminative mu/beta ERD over the sensorimotor strip and Broca
projections exactly on the paper's clinical ROI (Definition 4.1), so the
whole NAC pipeline (NAP, SPKD, CSI/SPR, conformal layer) can be executed
end-to-end and produce real numbers.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal

# --------------------------------------------------------------------------
# Montages and clinical ROI (Definition 4.1)
# --------------------------------------------------------------------------
KARAONE_CHANNELS = [
    # frontal (17)
    "Fp1", "Fp2", "Fpz", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    # fronto-central (9)
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT7", "FT8",
    # temporal (4)
    "T7", "T8", "TP7", "TP8",
    # central (7)
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    # centro-parietal (7)
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    # parietal (9)
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    # occipital / misc (11)
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz", "M1", "M2",
]

BCI2A_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Pz",
]

# Definition 4.1 (paper): C3, C4, Cz, F7, FC5, FC1, FC3, CP1, CP3
ROI_KARAONE = ["C3", "C4", "Cz", "F7", "FC5", "FC1", "FC3", "CP1", "CP3"]
ROI_BCI2A = ["C3", "C4", "Cz", "FC5", "FC1", "FC3", "CP1", "CP3"]  # F7 absent

# frequency bins: 0.5-45 Hz mapped onto n_freqs bins
FREQ_EDGES_DEFAULT = (0.5, 45.0)

KARAONE_CLASSES = ["a", "e", "i", "o", "u", "s", "sh", "pat", "pot", "knew", "gnaw"]
KARAONE_PHONEMES = KARAONE_CLASSES[:7]
KARAONE_WORDS = KARAONE_CLASSES[7:]

BCI2A_CLASSES = ["left_hand", "right_hand", "feet", "tongue"]


def roi_indices(channels, roi_names):
    """Indices of ROI channels inside a montage (case-insensitive)."""
    lookup = {c.upper(): i for i, c in enumerate(channels)}
    idx = [lookup[n.upper()] for n in roi_names if n.upper() in lookup]
    return np.array(idx, dtype=int)


def band_indices(n_freqs, lo, hi, edges=FREQ_EDGES_DEFAULT):
    """Indices of frequency bins inside [lo, hi] Hz."""
    freqs = np.linspace(edges[0], edges[1], n_freqs)
    return np.where((freqs >= lo) & (freqs <= hi))[0], freqs


# --------------------------------------------------------------------------
# Neurophysiology-plausible simulator
# --------------------------------------------------------------------------
def _smooth_profile(rng, n, width, center=None):
    """Gaussian-like bump over n positions."""
    x = np.arange(n)
    c = center if center is not None else rng.uniform(0.2, 0.8) * n
    w = width if width > 0 else n / 4.0
    return np.exp(-0.5 * ((x - c) / w) ** 2)


def _class_patterns(rng, channels, n_classes, n_freqs, n_times, roi_idx,
                    speech_like, snr):
    """Per-class spatio-spectral-temporal discriminative patterns.

    Patterns concentrate on ROI channels (motor strip + Broca) with a weak
    distractor on non-ROI channels, mimicking the experimental finding that
    task-discriminative spectral information lives over clinically defined
    cortices (paper, Section 1.3 / L1).
    """
    C = len(channels)
    patterns = []
    mu_f, _ = band_indices(n_freqs, 8, 12)
    beta_f, _ = band_indices(n_freqs, 13, 30)
    non_roi = np.setdiff1d(np.arange(C), roi_idx)

    for k in range(n_classes):
        # spatial weights: strong on ROI, weak distractors elsewhere
        w = np.zeros(C)
        n_roi_ch = rng.integers(max(4, len(roi_idx) // 2), len(roi_idx) + 1)
        sel = rng.choice(roi_idx, size=n_roi_ch, replace=False)
        w[sel] = rng.normal(0.8, 0.15, size=n_roi_ch)
        n_dis = min(3, len(non_roi))
        dis = rng.choice(non_roi, size=n_dis, replace=False)
        w[dis] = rng.normal(0.22, 0.08, size=n_dis)  # weaker distractors

        # spectral profile: mu for limb-like, beta for speech-like classes
        f_prof = np.zeros(n_freqs)
        band = beta_f if (speech_like and k % 2 == 1) or (not speech_like and k % 2 == 0) else mu_f
        for f in band:
            f_prof[f] += rng.normal(1.0, 0.2)
        # temporal profile
        t_prof = _smooth_profile(rng, n_times, n_times / 5.0)

        patterns.append((w, f_prof, t_prof))

    return patterns


def generate_dataset(dataset: str, cfg, seed: int = 0):
    """Simulate CWT-feature trials with class structure on the ROI.

    Returns dict with keys:
      X (N,C,F,T) float32, y (N,), subject (N,), channel names, class names,
      roi_idx (list[int])
    """
    rng = np.random.default_rng(seed)
    if dataset == "karaone":
        channels = KARAONE_CHANNELS
        classes = KARAONE_CLASSES
        speech_like = True
        n_subjects = 12 if cfg is None else cfg.n_subjects
        tps = 96 if cfg is None else cfg.trials_per_subject
        snr = 1.15
    elif dataset == "bci2a":
        channels = BCI2A_CHANNELS
        classes = BCI2A_CLASSES
        speech_like = False
        n_subjects = 9 if cfg is None else min(cfg.n_subjects, 9)
        tps = 96 if cfg is None else cfg.trials_per_subject
        snr = 1.7   # 4-class limb MI is more separable
    else:
        raise ValueError(dataset)

    C, F, T = len(channels), cfg.n_freqs, cfg.n_times
    K = len(classes)
    roi_idx = roi_indices(channels, ROI_KARAONE if dataset == "karaone" else ROI_BCI2A)
    mu_f, freqs = band_indices(F, 8, 12)
    beta_f, _ = band_indices(F, 13, 30)

    patterns = _class_patterns(rng, channels, K, F, T, roi_idx, speech_like, snr)

    # shared cue-locked motor ERD (all classes) over the sensorimotor strip
    motor_ch = roi_indices(channels, ["C3", "C4", "Cz", "CP1", "CP2"])
    mu_freq_prof = np.zeros(F)
    mu_freq_prof[mu_f] = 1.0
    beta_freq_prof = np.zeros(F)
    beta_freq_prof[beta_f] = 0.6
    cue_time = _smooth_profile(rng, T, T / 6.0, center=0.55 * T)

    X_list, y_list, subj_list = [], [], []
    for s in range(n_subjects):
        # subject idiosyncrasy: channel gains + noise level
        ch_gain = rng.lognormal(0.0, 0.12, size=C)
        noise_sd = rng.uniform(0.85, 1.15)
        base = 4.0 / (1.0 + 0.25 * np.arange(F))       # 1/f-like spectrum
        base = np.tile(base[None, :], (C, 1)) * ch_gain[:, None]

        for i in range(tps):
            y = i % K
            X = base[:, :, None] * np.ones((C, F, T))
            # shared ERD
            X[np.ix_(motor_ch, mu_f)] -= snr * 0.55 * cue_time[None, None, :]
            X[np.ix_(motor_ch, beta_f)] -= snr * 0.3 * cue_time[None, None, :]
            # class pattern
            w, f_prof, t_prof = patterns[y]
            pat = snr * (w[:, None, None] * f_prof[None, :, None] * t_prof[None, None, :])
            X = X + pat
            # noise + subject jitter
            X = X + rng.normal(0, noise_sd, size=X.shape)
            X = X + rng.normal(0, 0.05, size=(C, 1, 1))  # per-trial ch offset
            X_list.append(X.astype(np.float32))
            y_list.append(y)
            subj_list.append(s)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    subject = np.array(subj_list, dtype=np.int64)

    # z-score per channel per trial (paper preprocessing, last step)
    X = per_channel_zscore(X)

    return dict(X=X, y=y, subject=subject, channels=channels,
                classes=classes, roi_idx=roi_idx, freqs=freqs,
                mu_idx=mu_f, beta_idx=beta_f)


def per_channel_zscore(X, eps=1e-6):
    """z-score each trial/channel over (freq, time)."""
    mu = X.mean(axis=(1, 2), keepdims=True)
    sd = X.std(axis=(1, 2), keepdims=True)
    return (X - mu) / (sd + eps)


# --------------------------------------------------------------------------
# Split protocol
# --------------------------------------------------------------------------
def split_by_subject(data, train_frac=0.60, calib_frac=0.15, seed=0):
    """Pooled stratified split (train / calibration / test).

    The paper's primary protocol is cross-session with k=10 adaptation
    (train Session 1, adapt on 10 shots, test on remaining Session-2
    trials); `preset=paper` implements it with real data. The pooled
    split keeps the calibration/test roles (temperature scaling,
    conformal scores) intact at demo scale.
    """
    rng = np.random.default_rng(seed)
    X, y, subj = data["X"], data["y"], data["subject"]
    n = len(y)
    idx = np.arange(n)
    train_idx, calib_idx, test_idx = [], [], []
    for c in np.unique(y):
        ci = idx[y == c]
        rng.shuffle(ci)
        n_tr = int(train_frac * len(ci))
        n_ca = int(calib_frac * len(ci))
        train_idx.extend(ci[:n_tr])
        calib_idx.extend(ci[n_tr:n_tr + n_ca])
        test_idx.extend(ci[n_tr + n_ca:])
    rng.shuffle(train_idx)
    to_t = lambda a: torch.from_numpy(X[np.array(a)]).float()
    return dict(
        X_train=to_t(train_idx), y_train=torch.from_numpy(y[np.array(train_idx)]),
        X_calib=to_t(calib_idx), y_calib=torch.from_numpy(y[np.array(calib_idx)]),
        X_test=to_t(test_idx), y_test=torch.from_numpy(y[np.array(test_idx)]),
        subj_test=subj[np.array(test_idx)],
        idx=dict(train=np.array(train_idx), calib=np.array(calib_idx),
                 test=np.array(test_idx)),
    )


def to_tensors(data):
    return torch.from_numpy(data["X"]).float(), torch.from_numpy(data["y"])


# --------------------------------------------------------------------------
# Real-data loading utilities (used with --data-dir)
# --------------------------------------------------------------------------
def preprocess_raw(raw_eeg, fs, n_freqs=16, edges=FREQ_EDGES_DEFAULT, w0=6.0):
    """Paper preprocessing for raw multi-channel EEG (C, T_samples):
    0.5-45 Hz bidirectional Butterworth + Morlet CWT (w0=6) + z-score.
    Returns log-power CWT features (C, n_freqs, T_out).
    """
    b, a = signal.butter(4, [0.5 / (fs / 2), 45.0 / (fs / 2)], btype="band")
    filtered = signal.filtfilt(b, a, raw_eeg, axis=-1)
    widths = np.geomspace(edges[0], edges[1], num=n_freqs) * (2 * np.pi / (w0 * fs))
    cwt = signal.cwt(filtered[0], signal.morlet2, widths, w=w0)
    out = np.empty((filtered.shape[0], n_freqs, cwt.shape[1]), dtype=np.float32)
    for ch in range(filtered.shape[0]):
        out[ch] = np.abs(signal.cwt(filtered[ch], signal.morlet2, widths, w=w0))
    out = np.log1p(out)
    # resample time axis to a fixed length (e.g. 16 bins)
    t_out = 16
    if out.shape[-1] != t_out:
        out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1] // t_out + 1)[:,:,:,:t_out]
        out = out.mean(axis=3)
    out = per_channel_zscore(out)
    return out


def load_karaone_mat(path, n_freqs=16):
    """Load a KaraOne .mat export (fields: EEG per trial, labels).

    Expected layout (adjust to your export):
      mat['data']   : (N_trials, C, T) array at 1000 Hz
      mat['labels'] : (N_trials,) phoneme/word strings or ints
    """
    from scipy.io import loadmat
    mat = loadmat(path)
    data, labels = mat["data"], mat["labels"]
    feats = np.stack([preprocess_raw(t, fs=1000.0, n_freqs=n_freqs) for t in data])
    return feats, np.asarray(labels).astype(np.int64)


def load_bci2a_gdf(path, n_freqs=16):
    """Load a BCI IV-2a .gdf session (requires MNE)."""
    import mne  # optional dependency
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    raw.filter(0.5, 45.0, method="iir", verbose="ERROR")
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=4.0, picks=picks,
                        preload=True, verbose="ERROR")
    arr = epochs.get_data()
    feats = np.stack([preprocess_raw(t, fs=raw.info["sfreq"], n_freqs=n_freqs)
                      for t in arr])
    return feats, epochs.events[:, -1].astype(np.int64)
