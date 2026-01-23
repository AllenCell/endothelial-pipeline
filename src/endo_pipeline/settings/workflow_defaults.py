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

FIXED_SEG_FEATURE_MANIFEST_NAME: str = "fixed_merged_seg_features"
"""Default manifest name for merged CDH5 segmentation, CDH5 tracking and
NucViolet-stained nuclei segmentation features for fixed samples."""

DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS: str = "pca_reference"
"""Default dataset collection name for the segmentation feature workflow."""

DEFAULT_IMAGE_TYPE_FOR_SEMANTIC_CONDITIONING: Literal["bf", "cdh5"] = "bf"
"""Default image type to condition DiffAE models on."""

DIFFAE_IMAGE_LOADING_KEY_PREFIX: str = "raw_"
"""Default key prefix for loading DiffAE model input images."""

DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT: str = "raw_cdh5"
"""Default key for channel to use as diffusion image input to the model."""

DEFAULT_NUM_LATENT_DIMENSIONS: int = 1024
"""Default number of latent dimensions for DiffAE models."""

RANDOM_SEED: int = 47
"""Default random seed for workflows."""

MODEL_QC_NOISE_LEVELS: tuple = (0.25, 0.5, 0.75)
"""Default noise levels to add to ground truth for the model QC workflow."""

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
        "aspect_ratio",
        "cell_nuc_orientation_deg_rel_to_migration",
        "nuc_pos_rel_cell_angle_deg",
        "centroid_velocity_angle_deg",
        "cell_fluorescence_mean (a.u.)",
        "num_nuclei_in_crop",
        "area (um**2)",
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
