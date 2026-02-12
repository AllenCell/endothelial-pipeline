"""Workflow default settings."""

from typing import Literal

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

IMAGE_METRIC_DATASET_COLORS = {
    "validation_positions": "#2E86AB",  # Blue
    "rep_2_positions": "#A23B72",  # Purple/magenta
}

SEGMENTATION_FEATURE_COLUMNS = {
    "default": [
        "alignment_deg_rel_to_flow",
        "orientation_deg",
        "aspect_ratio",
        "centroid_velocity_angle_deg",
        "cell_fluorescence_mean (a.u.)",
        "num_nuclei_in_crop",
        "area (um**2)",
    ],
    "supp": [
        "alignment_deg_rel_to_flow",
        "orientation_deg",
        "orientation",
        "aspect_ratio",
        "cell_nuc_orientation_deg_rel_to_migration",
        "nuc_pos_rel_cell_angle_deg",
        "centroid_velocity_angle_deg",
        "cell_fluorescence_mean (a.u.)",
        "num_nuclei_in_crop",
        "area (um**2)",
    ],
    "dynamics_calculation_prereq": [
        "dataset_name",
        "position",
        "track_id",
        "time_minutes",
        "T",
        "centroid_X",
        "centroid_Y",
        "nuc_pos_rel_cell_X",
        "nuc_pos_rel_cell_Y",
        "pixel_size_xy_in_um",
        "alignment_deg_rel_to_flow",
        "nuc_pos_rel_cell_angle_deg",
        "cell_fluorescence_mean (a.u.)",
        "num_nuclei_in_crop",
    ],
    "filters": [
        "is_included",
        "is_greater_than_min_track_duration",
        "is_less_than_max_smoothed_area_normd_change",
        "is_edge_segmentation",
        "has_more_than_min_num_valid_points_per_track",
    ],
}
"""Name of segmentation features to include in analyses."""

DATASET_INFO_COLUMNS = [
    "dataset_name",
    "position",
    "image_index",
    "frame_number",
    "track_id",
    "crop_index",
    "label",
]
"""Name of dataset metadata columns required for analysis."""

# =========================================
# Default model configurations for model qc
# =========================================

# 10 models: 8 BF latent dims (8-1024) + 2 CDH5 conditioned (512, 1024)
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

# Order: 8 BF, 16 BF, 32 BF, 64 BF, 128 BF, 256 BF, 512 BF, 1024 BF, 512 CDH5, 1024 CDH5
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
"""Default x-axis labels for the 10-model latent dimension comparison bar plots."""
