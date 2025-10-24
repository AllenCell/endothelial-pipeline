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
