"""Workflow default settings."""

DEFAULT_MODEL_MANIFEST_NAME: str = "diffae_04_10"
"""Default model manifest for loading models and model features."""

DEFAULT_MODEL_RUN_NAME: str | None = None
"""Default model run name within the default model manifest."""

DEFAULT_PCA_DATASET_COLLECTION_NAME: str = "pca_reference"
"""Default dataset collection name for fitting PCA models."""

DEFAULT_SEG_FEATURE_MANIFEST_NAME: str = "live_merged_seg_features"
"""Default manifest name for merged CDH5 segmentation, CDH5 tracking and
label-free nuclei segmentation features."""

RANDOM_SEED: int = 47
"""Default random seed for workflows."""

MODEL_QC_DATASET_NAME = "20250224_20X"
"""Default dataset name for selecting crops for the model QC workflow."""

MODEL_QC_POSITION: int = 0
"""Default position index for selecting crops for the model QC workflow."""

MODEL_QC_TIMEPOINT: int = 0
"""Default timepoint index for selecting crops for the model QC workflow."""

MODEL_QC_NOISE_LEVELS: tuple = (0.25, 0.5, 0.75)
"""Default noise levels to add to ground truth for the model QC workflow."""

MODEL_QC_CROP_POSITION: tuple = (100, 100)
"""Default crop position (start_x, start_y) for the model QC workflow."""
