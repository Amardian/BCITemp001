"""Configuration and experimental presets.

All hyper-parameters follow the paper (Sections 4.3-4.7, 5.1-5.3):
  NAP : lambda=0.6, K=4 iterations, s=0.4 sparsity, fine-tune LR 1e-4
  QAT : b=8 bits, 500 calibration samples, per-channel weights /
        per-tensor activations, STE, fine-tune LR 5e-5
  SPKD: alpha=0.3, beta=0.3, T=4, Adam LR 1e-4, patience 10
  Expert layer: alpha_conformal=0.1 (90% coverage), tau_commit=0.7
  CSI clinical threshold: 0.85 ; non-inferiority margin delta = 1%

Three presets are provided:
  smoke : end-to-end correctness test (~2-4 min on a laptop CPU)
  demo  : reduced-scale faithful run (~30-60 min CPU)  -- DEFAULT
  paper : full paper-scale settings (all 5 seeds, full epochs, all 8
          orderings) -- needs the real datasets and a GPU, runs for days.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class Config:
    # ---------------- experiment ------------------------------------
    preset: str = "demo"
    seed: int = 42                     # paper seeds: 42,123,456,789,1024
    torch_threads: int = 2
    outdir: str = "results"
    datasets: tuple = ("karaone", "bci2a")
    device: str = "cpu"

    # ---------------- data (Section 5.1) ------------------------------
    # KaraOne : 12 subjects, 64 ch, 11 classes (7 phonemes + 4 words)
    # BCI IV2a:  9 subjects, 22 ch, 4 classes (limb MI)
    n_subjects: int = 12
    trials_per_subject: int = 96
    n_freqs: int = 16                  # CWT frequency bins (0.5-45 Hz)
    n_times: int = 16                  # CWT time bins per trial
    data_dir: str = ""                 # real-data directory (optional)

    # ---------------- model (Section 4.2) ------------------------------
    filters: tuple = (32, 64, 128, 256)   # 4 conv blocks
    kernel: int = 3                       # 3x3
    gru_hidden: int = 128                  # 2-layer GRU
    gru_layers: int = 2
    embedding_dim: int = 160               # per-channel embedding
    spatial_dim: int = 160                 # spatial readout width

    # ---------------- teacher training ---------------------------------
    batch: int = 32
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    teacher_epochs: int = 12

    # ---------------- NAP (Section 4.3.1 / Algorithm 1) -----------------
    nap_lambda: float = 0.6
    nap_iterations: int = 4
    nap_sparsity: float = 0.4
    nap_ft_epochs: int = 20
    nap_ft_lr: float = 1e-4
    nap_sds_trials: int = 16

    # ---------------- QAT (Section 4.3.2 / Algorithm 2) -----------------
    qat_bits: int = 8
    qat_calib: int = 500
    qat_epochs: int = 10
    qat_lr: float = 5e-5

    # ---------------- SPKD (Section 4.3.3 / Algorithm 3) ----------------
    spkd_alpha: float = 0.3
    spkd_beta: float = 0.3
    spkd_T: float = 4.0
    spkd_epochs: int = 30
    spkd_lr: float = 1e-4
    spkd_patience: int = 10
    spkd_batch: int = 32
    beta_soft: float = 1.0            # softplus sub-gradient beta (Eq. 11)

    # ---------------- baselines ----------------------------------------
    l1_ft_epochs: int = 20
    lth_epochs: int = 20
    baseline_epochs: int = 10

    # ---------------- XAI battery (Section 4.4) -------------------------
    ig_steps: int = 32
    n_null_shuffles: int = 1000
    dose_kmax: int = 10
    top_frac: float = 0.33             # Jaccard top-33%

    # ---------------- statistics (Section 5.2) ---------------------------
    n_bootstrap: int = 10000
    conformal_alpha: float = 0.1       # 90% coverage
    csi_threshold: float = 0.85
    noninf_margin: float = 0.01        # delta = 1%
    tau_commit: float = 0.7

    # ---------------- benchmarking (Section 4.7) -------------------------
    latency_warmup: int = 100
    latency_runs: int = 900
    sustained_runs: int = 1000
    window_len: int = 8                # sliding window (500 ms equiv.)
    window_step: int = 2               # 100 ms equiv.
    n_vote_windows: int = 5

    # ---------------- orderings / variants -------------------------------
    run_l1_baseline: bool = True
    run_lth: bool = True
    run_hinton: bool = True            # beta=0 variant (Table 16)
    run_alt_ordering: bool = True      # NAP->SPKD->QAT (Table 7)
    run_light_baselines: bool = True   # EEGNet / ShallowConvNet (Fig 5)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_preset(name: str, **overrides) -> Config:
    """Build a Config for one of the three presets, applying overrides."""
    base = dict(preset=name)

    if name == "smoke":
        base.update(
            datasets=("karaone",), n_subjects=4, trials_per_subject=24,
            batch=16, teacher_epochs=2,
            nap_iterations=1, nap_sparsity=0.2, nap_ft_epochs=1,
            nap_sds_trials=2, qat_calib=16, qat_epochs=1,
            spkd_epochs=1, spkd_patience=2, spkd_batch=16,
            l1_ft_epochs=1, lth_epochs=1, baseline_epochs=1,
            ig_steps=6, n_null_shuffles=40, dose_kmax=4,
            n_bootstrap=200, latency_warmup=5, latency_runs=25,
            sustained_runs=60, run_alt_ordering=False,
            run_light_baselines=False,
        )
    elif name == "demo":
        base.update(
            n_subjects=12, trials_per_subject=96, batch=32,
            teacher_epochs=12, lr=1.5e-3,
            # K=2 x 10% => 20% structured sparsity (paper: K=4 x 10% = 40%)
            nap_iterations=2, nap_sparsity=0.2, nap_ft_epochs=2,
            nap_sds_trials=8, qat_calib=128, qat_epochs=2,
            spkd_epochs=5, spkd_patience=10, spkd_batch=32,
            l1_ft_epochs=2, lth_epochs=8, baseline_epochs=3,
            ig_steps=24, n_null_shuffles=200, dose_kmax=10,
            n_bootstrap=1000, latency_warmup=20, latency_runs=200,
            sustained_runs=300,
        )
    elif name == "paper":
        base.update(
            n_subjects=12, trials_per_subject=None,   # full dataset
            batch=64, teacher_epochs=60,
            nap_iterations=4, nap_sparsity=0.4, nap_ft_epochs=20,
            nap_sds_trials=64, qat_calib=500, qat_epochs=10,
            spkd_epochs=30, spkd_patience=10, spkd_batch=64,
            l1_ft_epochs=20, lth_epochs=60, baseline_epochs=60,
            ig_steps=50, n_null_shuffles=1000, dose_kmax=10,
            n_bootstrap=10000, latency_warmup=100, latency_runs=900,
            sustained_runs=1800, run_alt_ordering=True,
        )
    else:
        raise ValueError(f"unknown preset {name!r}")

    base.update(overrides)
    cfg = Config(**{k: v for k, v in base.items() if k in Config.__dataclass_fields__})
    if cfg.preset != "paper":
        # demo/smoke run one seed; 'paper' repeats 5 seeds externally
        pass
    return cfg
