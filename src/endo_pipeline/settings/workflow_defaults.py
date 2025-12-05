"""Workflow default settings."""

from typing import Literal

DEFAULT_MODEL_MANIFEST_NAME: str = "diffae_baseline_exclude_cell_piling"
"""Default model manifest for loading models and model features."""

DEFAULT_MODEL_RUN_NAME: str | None = "20251110_latent_512"
"""Default model run name within the default model manifest."""

DEFAULT_PCA_DATASET_COLLECTION_NAME: str = "diffae_model_training"
"""Default dataset collection name for fitting PCA models."""

DEFAULT_SEG_FEATURE_MANIFEST_NAME: str = "live_merged_seg_features"
"""Default manifest name for merged CDH5 segmentation, CDH5 tracking and
label-free nuclei segmentation features."""

DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS: str = "pca_reference"
"""Default dataset collection name for the segmentation feature workflow."""

DEFAULT_IMAGE_TYPE_FOR_SEMANTIC_CONDITIONING: Literal["bf", "cdh5"] = "bf"
"""Default image type to condition DiffAE models on."""

DIFFAE_IMAGE_LOADING_KEY_PREFIX: str = "raw_"
"""Default key prefix for loading DiffAE model input images."""

DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT: str = "raw_cdh5"
"""Default key for channel to use as diffusion image input to the model."""

RANDOM_SEED: int = 47
"""Default random seed for workflows."""

MODEL_QC_NOISE_LEVELS: tuple = (0.25, 0.5, 0.75)
"""Default noise levels to add to ground truth for the model QC workflow."""

KERNEL_PARAMS_3D: dict = {
    "bandwidth": 0.145,
    "kernel": "gaussian",
}
"""Default kernel parameters for 3D flow field estimation."""

NUM_BINS_3D: tuple[int, int, int] = (50, 50, 50)
"""Default number of bins for 3D flow field estimation."""

TIME_STEP_IN_MINUTES: int = 5
"""Time step in minutes between consecutive time points for flow field estimation."""

INIT_POINT_3D: list = [0.0, -1.25, 0.75]
"""Default initial point for 3D flow field trajectory visualization."""

TRAJECTORY_TIME_SPAN: list[int] = [0, 5000]
"""Default time span for ODE solver in 3D flow field trajectory visualization."""

DATASET_COLLECTION_FOR_3D_DYNAMICS: str = "3d_flow_field_analysis"
"""Default dataset collection name for 3D dynamics analysis."""

OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS: str = "flow_field_3d"
"""Default output folder name for 3D dynamics analysis."""

TRAJECTORY_DICT_FILE_NAME: str = "traj_dict"
"""Default file name for saving trajectory dictionaries."""
