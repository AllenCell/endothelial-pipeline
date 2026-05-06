"""Model QC package for Diffusion Autoencoder evaluation.

This package provides helpers for:

- Loading and preprocessing images for QC (:mod:`.image_loading`)
- Running denoising experiments (:mod:`.denoising`)
- Saving crops as TIFF files (:mod:`.tiff_io`)
- Generating contact sheets and summary figures (:mod:`.plotting`)
- Image similarity metrics (correlation, SSIM, LPIPS) (:mod:`.image_metrics`)
- Computing and aggregating evaluation metrics (:mod:`.metrics`)
- Orchestrating single-model evaluation (:mod:`.evaluation`)
"""

from .denoising import run_denoising_experiments
from .evaluation import ModelKey, evaluate_single_model
from .image_loading import load_and_preprocess_example_crop, load_transformed_image
from .image_metrics import (
    ImageMetrics,
    LPIPSCalculator,
    compute_all_metrics,
    compute_correlation,
    compute_denoising_metrics,
    compute_ssim,
)
from .metrics import (
    aggregate_seed_metrics,
    build_models_data,
    compute_baseline_data,
    compute_baseline_for_example,
    create_comparison_plots_and_summary,
)
from .plotting import (
    save_intermediate_contact_sheet,
    save_negative_control_sheet,
    save_summary_figure,
)
from .tiff_io import save_denoising_crops, save_image_as_tiff

__all__ = [
    "ImageMetrics",
    "LPIPSCalculator",
    "ModelKey",
    "aggregate_seed_metrics",
    "build_models_data",
    "compute_all_metrics",
    "compute_baseline_data",
    "compute_baseline_for_example",
    "compute_correlation",
    "compute_denoising_metrics",
    "compute_ssim",
    "create_comparison_plots_and_summary",
    "evaluate_single_model",
    "load_and_preprocess_example_crop",
    "load_transformed_image",
    "run_denoising_experiments",
    "save_denoising_crops",
    "save_image_as_tiff",
    "save_intermediate_contact_sheet",
    "save_negative_control_sheet",
    "save_summary_figure",
]
