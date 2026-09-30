"""NAC: Neurophysiology-Aware Compression for edge-deployable EEG-SMI BCIs.

Complete Python reimplementation of:
  "Edge-Deployable Explainable EEG-Based Speech Motor Imagery Classification
   via Neurophysiology-Aware Compression and Causal Saliency Validation"

Pipeline (paper Section 4):
  Stage 1  preprocessing (bandpass 0.5-45 Hz, ICA, Morlet CWT, z-score)
  Stage 2  CWT time-frequency features  X in R^{C x F x T}
  Stage 3  NAC compression  =  NAP -> INT8 QAT -> SPKD
  Stage 4  temperature-scaling calibration
  Stage 5  conformal prediction with abstention (90% nominal coverage)
  Stage 6  clinical decision rule (tau_commit = 0.7)
"""

__version__ = "1.0.0"

from .config import Config, get_preset  # noqa: F401
