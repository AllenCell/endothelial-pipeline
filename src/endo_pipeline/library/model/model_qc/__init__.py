"""Model QC package for Diffusion Autoencoder evaluation.

This package provides helpers for:

- Loading and preprocessing images for QC (:mod:`.image_loading`)
- Running denoising experiments (:mod:`.denoising`)
- Saving crops as TIFF files (:mod:`.tiff_io`)
- Generating contact sheets and summary figures (:mod:`.plotting`)
- Image similarity metrics (correlation, SSIM, LPIPS) (:mod:`.image_metrics`)
- Computing and aggregating evaluation metrics (:mod:`.metrics`)
- Orchestrating single-model evaluation (:mod:`.evaluation`)
- Persisting per-model parquets and loading them via dataframe manifests
  (:mod:`.results_io`)
"""

from .denoising import run_denoising_experiments
from .evaluation import ModelKey
from .metrics import aggregate_seed_metrics, build_models_data, compute_baseline_data

__all__ = [
    "ModelKey",
    "aggregate_seed_metrics",
    "build_models_data",
    "compute_baseline_data",
    "run_denoising_experiments",
]
