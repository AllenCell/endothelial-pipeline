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
"""Matplotlib text-box properties for metric annotations in QC plots."""

IMAGE_METRIC_DATASET_COLORS = {
    "validation_positions": "#2C6FAC",  # Medium blue
    "rep_2_positions": "#7ABBE0",  # Light blue
}
"""Color palette for image metric bar plots, keyed by dataset split name."""

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
        "label",
        "time_minutes",
        "T",
        "centroid_X",
        "centroid_Y",
        "nuc_pos_rel_cell_X",
        "nuc_pos_rel_cell_Y",
        "pixel_size_xy_in_um",
        "time_resolution_minutes",
        "alignment_deg_rel_to_flow",
        "nuc_pos_rel_cell_angle_deg",
        "cell_fluorescence_mean (a.u.)",
        "num_nuclei_in_crop",
        "all_labels_in_crop",
    ],
    "filters": [
        "is_included",
        "is_greater_than_min_track_duration",
        "is_less_than_max_smoothed_area_normd_change",
        "is_edge_segmentation",
        "has_more_than_min_num_valid_points_per_track",
        "bbox_is_in_bounds",
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

DEFAULT_OPTICAL_FLOW_COLLECTION: str = "diffae_model_training"
"""Default dataset collection for the optical-flow feature workflow."""

DEFAULT_OPTICAL_FLOW_MANIFEST_NAME: str = "optical_flow_bf"
"""Default dataframe manifest name for optical-flow features."""

OPTICAL_FLOW_BASE_FEATURES: list = [
    "optical_flow_mean_speed",
    "optical_flow_mean_unit_vector",
    "optical_flow_std_speed",
    "optical_flow_mean_angle",
    "optical_flow_angle_std",
    "optical_flow_mean_u",
    "optical_flow_mean_v",
    "optical_flow_std_u",
    "optical_flow_std_v",
]
"""Base feature names computed per (crop, timepoint, dt) by optical-flow extraction."""

OPTICAL_FLOW_CHANNEL_PERCENTILE: dict = {
    "EGFP": 95,  # sparse fluorescence → exclude empty background
    "BF": 0,  # dense texture → keep all pixels
}
"""Intensity percentile thresholds per channel for optical-flow masking."""

OPTICAL_FLOW_CHANNEL_ATTACHMENT: dict[str, float] = {
    "EGFP": 7.5,  # half of skimage default (15); wider normalised range
    "BF": 2.5,    # z-score normalisation compresses dynamic range
}
"""TVL1 attachment (lambda) per channel.

Smaller values produce smoother flow fields.  BF uses a lower value
because z-score normalisation compresses the intensity range, so a
weaker data-fidelity term avoids fitting noise."""

DEFAULT_OPTICAL_FLOW_MAX_DT: int = 1
"""Maximum frame gap for multi-scale optical-flow sweep (dt = 1 ... MAX_DT)."""

DEFAULT_OMP_NUM_THREADS: str = "1"
"""Default OMP_NUM_THREADS for optical-flow workers (pinned to avoid over-subscription)."""

DEFAULT_OPENBLAS_NUM_THREADS: str = "1"
"""Default OPENBLAS_NUM_THREADS for optical-flow workers."""

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

OPTICAL_FLOW_BASE_FEATURES: list = [
    "optical_flow_mean_speed_dt1",
    "optical_flow_mean_unit_vector_dt1",
    "optical_flow_std_speed_dt1",
    "optical_flow_mean_angle_dt1",
    "optical_flow_angle_std_dt1",
    "optical_flow_mean_u_dt1",
    "optical_flow_mean_v_dt1",
    "optical_flow_std_u_dt1",
    "optical_flow_std_v_dt1",
]
"""Base feature names computed per (crop, timepoint, dt) by optical-flow extraction."""
