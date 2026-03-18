"""Workflow default settings."""

from typing import Literal

from endo_pipeline.settings.segmentation_feature_dataframes import ColumnNameSeg as ColNmSeg

DEFAULT_MODEL_MANIFEST_NAME: str = "diffae_baseline_exclude_cell_piling"
"""Default model manifest for loading models and model features."""

DEFAULT_MODEL_RUN_NAME: str = "20251110_latent_512"
"""Default model run name within the default model manifest."""

DEFAULT_PCA_DATASET_COLLECTION_NAME: str = "diffae_model_training"
"""Default dataset collection name for fitting PCA models."""

DEFAULT_SEG_FEATURE_MANIFEST_NAME: str = "live_merged_seg_features"
"""Default manifest name for merged CDH5 segmentation, CDH5 tracking and
label-free nuclei segmentation features."""

DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME: str = "pc_diffae_tracked_seg_features"
"""Default manifest name for PCA-reduced DiffAE tracked-cell features merged with
DiffAE tracked-cell features and CDH5 segmentation features."""

FIXED_SEG_FEATURE_MANIFEST_NAME: str = "fixed_merged_seg_features"
"""Default manifest name for merged CDH5 segmentation, CDH5 tracking and
NucViolet-stained nuclei segmentation features for fixed samples."""

DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS: str = "pca_reference"
"""Default dataset collection name for the segmentation feature workflow."""

DEFAULT_IMAGE_TYPE_FOR_SEMANTIC_CONDITIONING: Literal["bf", "cdh5"] = "bf"
"""Default image type to condition DiffAE models on."""

DIFFAE_IMAGE_LOADING_KEY_PREFIX: str = "raw_"
"""Default key prefix for loading DiffAE model input images."""

DIFFAE_EVAL_DATAFRAME_MANIFEST_PREFIX: str = "diffae_evaluation_dataframe_"
"""Prefix for DiffAE model evaluation image loading dataframe."""

DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT: str = "raw_cdh5"
"""Default key for channel to use as diffusion image input to the model."""

DEFAULT_NUM_LATENT_DIMENSIONS: int = 1024
"""Default number of latent dimensions for DiffAE models."""

RANDOM_SEED: int = 47
"""Default random seed for workflows."""

MODEL_QC_NOISE_LEVELS: tuple = (0.25, 0.5, 0.75)
"""Default noise levels to add to ground truth for the model QC workflow."""

METRIC_TEXT_BOX_PROPS = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3}
"""Matplotlib text-box properties for metric annotations in QC plots."""

IMAGE_METRIC_DATASET_COLORS = {
    "validation_positions": "#2C6FAC",  # Medium blue
    "rep_2_positions": "#7ABBE0",  # Light blue
}
"""Color palette for image metric bar plots, keyed by dataset split name."""

SEGMENTATION_FEATURE_COLUMNS = {
    "default": [
        ColNmSeg.ALIGNMENT_DEG,
        ColNmSeg.ORIENTATION_DEG,
        ColNmSeg.ASPECT_RATIO,
        ColNmSeg.CENTROID_VELOCITY_ANGLE_DEG,
        ColNmSeg.CELL_FLUOR_MEAN,
        ColNmSeg.EDGE_FLUOR_MEAN,
        ColNmSeg.NODE_FLUOR_MEAN,
        ColNmSeg.NUM_NUCLEI_IN_CROP,
        ColNmSeg.AREA_UM_SQ,
    ],
    "supp": [
        ColNmSeg.ALIGNMENT_DEG,
        ColNmSeg.ORIENTATION_DEG,
        ColNmSeg.ORIENTATION,
        ColNmSeg.ASPECT_RATIO,
        ColNmSeg.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG,
        ColNmSeg.NUCLEI_POSITION_ANGLE_DEG,
        ColNmSeg.CENTROID_VELOCITY_ANGLE_DEG,
        ColNmSeg.CELL_FLUOR_MEAN,
        ColNmSeg.EDGE_FLUOR_MEAN,
        ColNmSeg.NODE_FLUOR_MEAN,
        ColNmSeg.NUM_NUCLEI_IN_CROP,
        ColNmSeg.AREA_UM_SQ,
    ],
    "dynamics_calculation_prereq": [
        ColNmSeg.DATASET,
        ColNmSeg.POSITION,
        ColNmSeg.TRACK_ID,
        ColNmSeg.LABEL,
        ColNmSeg.TIME_HRS,
        ColNmSeg.TIME_MINS,
        ColNmSeg.TIMEPOINT,
        ColNmSeg.CENTROID_X,
        ColNmSeg.CENTROID_Y,
        ColNmSeg.NUCLEI_POSITION_X,
        ColNmSeg.NUCLEI_POSITION_Y,
        ColNmSeg.PIXEL_SIZE_XY_IN_UM,
        ColNmSeg.TIME_RESOLUTION_MINUTES,
        ColNmSeg.ALIGNMENT_DEG,
        ColNmSeg.NUCLEI_POSITION_ANGLE_DEG,
        ColNmSeg.CELL_FLUOR_MEAN,
        ColNmSeg.EDGE_FLUOR_MEAN,
        ColNmSeg.NODE_FLUOR_MEAN,
        ColNmSeg.NUM_NUCLEI_IN_CROP,
        ColNmSeg.LABELS_IN_CROP,
    ],
    "filters": [
        ColNmSeg.IS_INCLUDED,
        ColNmSeg.IS_GREATER_THAN_MIN_TRACK_DURATION,
        ColNmSeg.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE,
        ColNmSeg.IS_EDGE_SEGMENTATION,
        ColNmSeg.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK,
        ColNmSeg.IS_VALID_BBOX,
    ],
}
"""Name of segmentation features to include in analyses."""

DATASET_INFO_COLUMNS = [
    ColNmSeg.DATASET,
    ColNmSeg.POSITION,
    ColNmSeg.TIMEPOINT,
    ColNmSeg.TRACK_ID,
    ColNmSeg.CROP_INDEX,
    ColNmSeg.LABEL,
]
"""Name of dataset metadata columns required for analysis."""

# =========================================
# Default model configurations for model qc
# =========================================

DEFAULT_MODEL_QC_MANIFEST_NAMES = [
    "diffae_baseline_exclude_cell_piling",  # 8 BF
    "diffae_baseline_exclude_cell_piling",  # 16 BF
    "diffae_baseline_exclude_cell_piling",  # 32 BF
    "diffae_baseline_exclude_cell_piling",  # 64 BF
    "diffae_baseline_exclude_cell_piling",  # 128 BF
    "diffae_baseline_exclude_cell_piling",  # 256 BF
    "diffae_baseline_exclude_cell_piling",  # 512 BF
    "diffae_baseline_exclude_cell_piling",  # 1024 BF
    "diffae_cdh5_conditioned",  # 512 CDH5
    "diffae_cdh5_conditioned",  # 1024 CDH5
]
"""Default manifest names for the 10-model QC comparison study.

Covers 8 brightfield-conditioned latent dimensions (8--1024) and 2 CDH5-
conditioned models (512, 1024).
"""

DEFAULT_MODEL_QC_RUN_NAMES = [
    "20260207_latent_8",
    "20260205_latent_16",
    "20260203_latent_32",
    "20260206_latent_64",
    "20260127_latent_128",
    "20260122_latent_256",
    "20251110_latent_512",
    "20251110_latent_1024",
    "20260130_latent_512",
    "20251110_latent_1024",
]
"""Run names corresponding to each entry in :data:`DEFAULT_MODEL_QC_MANIFEST_NAMES`."""
DEFAULT_MODEL_QC_LABELS = [
    "8 BF",
    "16 BF",
    "32 BF",
    "64 BF",
    "128 BF",
    "256 BF",
    "512 BF",
    "1024 BF",
    "512 CDH5",
    "1024 CDH5",
]
"""X-axis labels for the 10-model latent dimension comparison bar plots.

Order: 8 BF, 16 BF, 32 BF, 64 BF, 128 BF, 256 BF, 512 BF, 1024 BF,
512 CDH5, 1024 CDH5.
"""
