"""Workflow default settings."""

from typing import Literal

from endo_pipeline.configs.dataset_config import TimepointAnnotation
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType

DEFAULT_MODEL_MANIFEST_NAME: str = "diffae_baseline"
"""Default model manifest for loading models and model features."""

DEFAULT_MODEL_RUN_NAME: str = "latent_512"
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

FEATURES_UNFILTERED_MANIFEST_NAMES: dict[Literal["grid", "tracked"], str] = {
    "grid": GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME,
    "tracked": CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
}
"""Mapping of crop pattern to unfiltered feature dataframe manifest name."""

FEATURES_FILTERED_MANIFEST_NAMES: dict[Literal["grid", "tracked"], str] = {
    "grid": GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    "tracked": CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
}
"""Mapping of crop pattern to filtered feature dataframe manifest name."""

DEFAULT_IMAGE_TYPE_FOR_SEMANTIC_CONDITIONING: Literal["bf", "cdh5"] = "bf"
"""Default image type to condition DiffAE models on."""

DIFFAE_IMAGE_LOADING_KEY_PREFIX: str = "raw_"
"""Default key prefix for loading DiffAE model input images."""

DIFFAE_TRAIN_DATAFRAME_MANIFEST_PREFIX: str = "diffae_training_dataframe"
"""Prefix for DiffAE model training image loading dataframe."""

DIFFAE_EVAL_DATAFRAME_MANIFEST_PREFIX: str = "diffae_evaluation_dataframe"
"""Prefix for DiffAE model evaluation image loading dataframe."""

DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT: str = "raw_cdh5"
"""Default key for channel to use as diffusion image input to the model."""

DEFAULT_NUM_LATENT_DIMENSIONS: int = 512
"""Default number of latent dimensions for DiffAE models."""

RANDOM_SEED: int = 47
"""Default random seed for workflows."""

MODEL_QC_NOISE_LEVELS: tuple = (0.25, 0.5, 0.75)
"""Default noise levels to add to ground truth for the model QC workflow."""

LABELFREE_NUCLEI_MODEL_MANIFEST_NAME: str = "nuc_pred_labelfree"
"""Default manifest name for the label-free nuclei segmentation model."""

LABELFREE_NUCLEI_MODEL_RUN_NAME: str = "finetuned_20250419"
"""Default run name for the label-free nuclei segmentation model."""

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

DEFAULT_MODEL_QC_LABEL_MAP: dict[tuple[str, str], str] = {
    ("diffae_baseline", "latent_8"): "BF-8",
    ("diffae_baseline", "latent_16"): "BF-16",
    ("diffae_baseline", "latent_32"): "BF-32",
    ("diffae_baseline", "latent_64"): "BF-64",
    ("diffae_baseline", "latent_128"): "BF-128",
    ("diffae_baseline", "latent_256"): "BF-256",
    ("diffae_baseline", "latent_512"): "BF-512",
    ("diffae_baseline", "latent_1024"): "BF-1024",
    ("diffae_cdh5_conditioned", "latent_512"): "VE-cad-\n512",
    ("diffae_cdh5_conditioned", "latent_1024"): "VE-cad-\n1024",
}
"""Map of `(manifest_name, run_name) to bar plot label."""

DEFAULT_MODEL_COMPARISON_RUNS: list[tuple[str, str]] = list(DEFAULT_MODEL_QC_LABEL_MAP.keys())
"""List of default model comparison model runs (same order as label map)."""

# Parallel lists derived from the label map (the single source of truth) so the
# manifest / run / label ordering always stays in sync.
DEFAULT_MODEL_QC_MANIFEST_NAMES: list[str] = [
    manifest for manifest, _ in DEFAULT_MODEL_QC_LABEL_MAP
]
"""Manifest name for each model in the QC comparison sweep, in plot order."""

DEFAULT_MODEL_QC_RUN_NAMES: list[str] = [run for _, run in DEFAULT_MODEL_QC_LABEL_MAP]
"""Run name for each model in the QC comparison sweep, in plot order."""

DEFAULT_MODEL_QC_LABELS: list[str] = list(DEFAULT_MODEL_QC_LABEL_MAP.values())
"""Bar-plot label for each model in the QC comparison sweep, in plot order."""

DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX: str = "diffae_model_comparison_metrics"
"""Prefix for the dataframe manifests cataloguing the per-model QC parquets.

The production workflow emits one dataframe manifest per
``manifest_name`` in the curated sweep, named
``{PREFIX}_{manifest_name}`` (with a ``_demo`` suffix when run in demo
mode). Each manifest's ``locations`` dict is keyed by ``run_name`` and
points at the per-(manifest, run) metrics parquet. The companion plot
workflow loads all of these manifests and renders the supplementary
bar chart without re-running inference.
"""
