from enum import StrEnum

"""Global constants and default settings for DiffAE feature dataframe creation and processing."""


class ColumnName(StrEnum):
    """Dataframe column names used in DiFFAE feature dataframes."""

    LATENT_FEATURE_PREFIX = "feat_"
    """Prefix for latent feature column names."""

    PCA_FEATURE_PREFIX = "pc_"
    """Prefix for PCA-transformed feature column names."""

    DATASET = "dataset"
    """Column name for dataset name."""

    MODEL_MANIFEST = "model_manifest_name"
    """Column name for model manifest name."""

    MODEL_RUN = "run_name"
    """Column name for model run name."""

    POSITION = "position"
    """Column name for position identifier."""

    TIMEPOINT = "frame_number"
    """Column name for timepoint (frame number)."""

    ZARR_PATH = "zarr_path"
    """Column name for path to the source Zarr file."""

    RESOLUTION = "resolution_level"
    """Column name for resolution level of the image."""

    CROP_INDEX = "crop_index"
    """Column name for crop index."""

    START_X = "start_x"
    """Upper-left x-coordinate of the crop."""

    START_Y = "start_y"
    """Upper-left y-coordinate of the crop."""

    END_X = "end_x"
    """Lower-right x-coordinate of the crop."""

    END_Y = "end_y"
    """Lower-right y-coordinate of the crop."""

    CROP_SIZE_X = "crop_size_x"
    """Width of the crop in pixels."""

    CROP_SIZE_Y = "crop_size_y"
    """Height of the crop in pixels."""


class CytoDLLoadDataKeys(StrEnum):
    """Metadata keys passed into MultiDimImageDataset for loading images."""

    FILE_PATH = "path"
    """Key for path to the source image file."""

    TIMEPOINT = "T"
    """Key for timepoint (frame number)."""

    TIME_START = "frame_start"
    """Key for starting timepoint (frame number)."""

    TIME_END = "frame_stop"
    """Key for ending timepoint (frame number)."""

    TIME_STEP = "frame_step"
    """Key for timepoint (frame number) step size."""

    INCLUDE_TIMEPOINTS = "include_frames"
    """Key for list of timepoints (frame numbers) to include."""

    Z_START = "z_start"
    """Key for starting z-slice index."""

    Z_END = "z_stop"
    """Key for ending z-slice index."""

    Z_STEP = "z_step"
    """Key for z-slice index step size."""

    CHANNELS = "channel"
    """Key for list of channels to load."""

    RESOLUTION = "resolution_level"
    """Key for resolution level of the image."""

    SCENE = "scene"
    """Key for scene identifier."""


class CytoDLSaveDataKeys(StrEnum):
    """Metadata keys passed to CytoDL callback object SaveTabularData."""

    FILE_PATH = "filename_or_obj"
    """Key for path to the source image file."""

    TIMEPOINT = "T"
    """Key for timepoint (frame number)."""


NUM_LATENT_FEATURES = 8
"""Number of latent features to extract from DiFFAE model."""

NUM_PCS_TO_ANALYZE = 3
"""Number of top principal components to analyze."""

DIFFAE_FEATURE_COLUMN_NAMES = [
    f"{ColumnName.LATENT_FEATURE_PREFIX}{i}" for i in range(NUM_LATENT_FEATURES)
]
"""Full set of column names for original latent features in DiFFAE feature dataframes."""

DIFFAE_PC_COLUMN_NAMES = [
    f"{ColumnName.PCA_FEATURE_PREFIX}{i+1}" for i in range(NUM_LATENT_FEATURES)
]
"""Full set of column names for PCA-transformed features in DiFFAE feature dataframes."""
