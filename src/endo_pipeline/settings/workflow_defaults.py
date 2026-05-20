"""Workflow default settings."""

from typing import Literal

from endo_pipeline.configs.dataset_config import TimepointAnnotation
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType

DEFAULT_MODEL_MANIFEST_NAME: str = "diffae_baseline_exclude_cell_piling"
"""Default model manifest for loading models and model features."""

DEFAULT_MODEL_RUN_NAME: str = "20251110_latent_512"
"""Default model run name within the default model manifest."""

DEFAULT_PCA_DATASET_COLLECTION_NAME: str = "diffae_model_training"
"""Default dataset collection name for fitting PCA models."""

DEFAULT_SEG_FEATURE_MANIFEST_NAME: str = "merged_segmentation_features"
"""Default manifest name for merged CDH5 segmentation, CDH5 tracking and
label-free nuclei segmentation features."""

DIFFAE_PCA_FEATURE_TRACKED_UNFILTERED_MANIFEST_NAME: str = "diffae_pca_features_tracked_unfiltered"
"""Dataframe manifest name for unfiltered PCA-reduced DiffAE features for track-based crops."""

DIFFAE_PCA_FEATURE_TRACKED_FILTERED_MANIFEST_NAME: str = "diffae_pca_features_tracked_filtered"
"""Dataframe manifest name for filtered PCA-reduced DiffAE features for track-based crops."""

GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME: str = "grid_based_features_unfiltered"
"""Dataframe manifest name for unfiltered grid-based features."""

GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME: str = "grid_based_features_filtered"
"""Dataframe manifest name for filtered grid-based features."""

CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME: str = "cell_centered_features_unfiltered"
"""Dataframe manifest name for unfiltered cell-centered features."""

CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME: str = "cell_centered_features_filtered"
"""Dataframe manifest name for unfiltered cell-centered features."""

DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS: str = "pca_reference"
"""Default dataset collection name for the segmentation feature workflow."""

FIRST_PASSAGE_TIME_MANIFEST_NAME: str = "first_passage_time_statistics"
"""Manifest name for first passage time statistics dataframe."""

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

SEGMENTATION_FEATURE_COLUMNS: dict[str, list[ColumnNameType]] = {
    "default": [
        Column.SegData.ALIGNMENT_DEG,
        Column.SegData.ORIENTATION,
        Column.SegData.ASPECT_RATIO,
        Column.SegData.CENTROID_VELOCITY_ANGLE_DEG,
        Column.SegData.CELL_FLUOR_MEAN,
        Column.SegData.EDGE_FLUOR_MEAN,
        Column.SegData.NODE_FLUOR_MEAN,
        Column.SegData.NUM_NUCLEI_IN_CROP,
        Column.SegData.AREA_UM_SQ,
    ],
    "main_figure": [
        Column.SegData.ORIENTATION,
        Column.SegData.ASPECT_RATIO,
        Column.SegData.NUM_NUCLEI_IN_CROP,
        Column.SegData.AREA_UM_SQ,
        Column.SegData.CELL_FLUOR_MEAN,
        Column.SegData.EDGE_FLUOR_MEAN,
        Column.OpticalFlow.UNIT_VECTOR_MEAN,
        Column.OpticalFlow.SPEED_MEAN,
    ],
    "supp_figure": [
        Column.SegData.ORIENTATION,
        Column.SegData.ALIGNMENT_DEG,
        Column.SegData.ASPECT_RATIO,
        Column.SegData.NUM_NUCLEI_IN_CROP,
        Column.SegData.AREA_UM_SQ,
        Column.SegData.CELL_FLUOR_MEAN,
        Column.SegData.EDGE_FLUOR_MEAN,
        Column.OpticalFlow.UNIT_VECTOR_MEAN,
        Column.OpticalFlow.SPEED_MEAN,
        Column.SegData.CENTROID_VELOCITY_UM_PER_MIN,
        Column.SegData.CENTROID_VELOCITY_ANGLE_DEG,
        Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG,
        Column.SegData.NUCLEI_POSITION_ANGLE_DEG,
    ],
    "dynamics_calculation_prereq": [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.TRACK_ID,
        Column.PIXEL_SIZE_XY_IN_UM,
        Column.TIME_RESOLUTION_MINUTES,
        Column.SegData.LABEL,
        Column.SegData.TIME_HRS,
        Column.SegData.TIME_MINS,
        Column.SegData.CENTROID_X,
        Column.SegData.CENTROID_Y,
        Column.SegData.NUCLEI_POSITION_X,
        Column.SegData.NUCLEI_POSITION_Y,
        Column.SegData.ALIGNMENT_DEG,
        Column.SegData.NUCLEI_POSITION_ANGLE_DEG,
        Column.SegData.CELL_FLUOR_MEAN,
        Column.SegData.EDGE_FLUOR_MEAN,
        Column.SegData.NODE_FLUOR_MEAN,
        Column.SegData.NUM_NUCLEI_IN_CROP,
        Column.SegData.LABELS_IN_CROP,
    ],
    "filters": [
        Column.SegDataFilters.IS_INCLUDED,
        Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION,
        Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE,
        Column.SegDataFilters.IS_EDGE_SEGMENTATION,
        Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK,
        Column.SegDataFilters.IS_VALID_BBOX,
    ],
}
"""Name of segmentation features to include in analyses."""

DEFAULT_COLUMNS_TO_DROP: dict[str, list[ColumnNameType]] = {
    "segmentation_features": [
        Column.SegData.EDGE_FLUOR,
        Column.SegData.NODE_FLUOR,
        Column.SegData.CELL_FLUOR_MEDIAN,
        Column.SegData.CELL_FLUOR_MAX,
        Column.SegData.CELL_FLUOR_MIN,
        Column.SegData.CELL_FLUOR_PCT25,
        Column.SegData.CELL_FLUOR_PCT75,
        Column.SegData.RESOLUTION_FOR_DIFFAE,
    ],
    "segmentation_filters": [
        Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF,
        Column.SegDataFilters.MIN_TRACK_DURATION,
        Column.SegDataFilters.MAX_SMOOTHED_AREA_NORMALIZED_CHANGE,
        Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK,
        Column.SegDataFilters.MIN_NUM_VALID_TIMEPOINTS_PER_TRACK,
    ],
    "verification_columns": [
        Column.SegDataWorkflowVerification.SEGMENTATION_PATH,
        Column.SegDataWorkflowVerification.TRACKING_REF_IDX,
        Column.SegDataWorkflowVerification.TRACKING_MATCHED_QUERY_LABEL,
        Column.SegDataWorkflowVerification.TRACKING_OPTIMIZED_METRIC_VAL,
        Column.SegDataWorkflowVerification.TRACKING_MATCHING_METHOD,
        Column.SegDataWorkflowVerification.NUM_NUC_WITH_MOST_OVERLAP,
        Column.SegDataWorkflowVerification.SMOOTHED_AREA_NORMALIZED,
        Column.SegDataWorkflowVerification.SIGMA_FOR_AREA_SMOOTHING,
        Column.SegDataWorkflowVerification.NUM_UNIQUE_TRACKS_PER_TIMEPOINT,
        Column.SegDataWorkflowVerification.NODE_LABELS,
        Column.SegDataWorkflowVerification.EDGE_LABELS,
        Column.SegDataWorkflowVerification.NODE_PAIR_LABELS,
        Column.SegDataWorkflowVerification.NUCLEI_LABELS_IN_CDH5_SEGMENTATION,
        Column.SegDataWorkflowVerification.NUCLEI_FRACTION_IN_CDH5_SEGMENTATION,
    ],
    "diffae_columns": [
        Column.DiffAEData.MODEL_MANIFEST,
        Column.DiffAEData.MODEL_RUN,
        Column.DiffAEData.CROP_SIZE_X,
        Column.DiffAEData.CROP_SIZE_Y,
        Column.DiffAEData.RESOLUTION,
    ],
    "base_columns": [
        Column.CDH5_CHANNEL_INDEX_ZARR,
        Column.BF_CHANNEL_INDEX_ZARR,
    ],
}

ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS = [
    TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
    TimepointAnnotation.GFP_SCOPE_ERROR,
]
"""Timepoint annotations that apply to segmentation feature datasets."""

DATASET_INFO_COLUMNS: list[ColumnNameType] = [
    Column.DATASET,
    Column.POSITION,
    Column.TIMEPOINT,
    Column.TRACK_ID,
    Column.CROP_INDEX,
    Column.SegData.LABEL,
]
"""Name of dataset metadata columns required for analysis."""

# =========================================
# Default model configurations for model qc
# =========================================

DEFAULT_MODEL_QC_MANIFEST_NAMES: list[str] = [
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

DEFAULT_MODEL_QC_RUN_NAMES: list[str] = [
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
DEFAULT_MODEL_QC_LABELS: list[str] = [
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
