from enum import StrEnum

from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate

"""Global constants and default settings for DiffAE feature dataframe creation and processing."""


class CytoDLLoadDataKeys(StrEnum):
    """Metadata keys passed into MultiDimImageDataset for loading images."""

    FILE_PATH = Column.SOURCE_IMAGE_PATH_FOR_MODEL
    """Key for path to the source image file."""

    TIMEPOINT = Column.TIMEPOINT_FOR_MODEL
    """Key for timepoint (frame number)."""

    TIME_START = Column.FRAME_START_FOR_MODEL
    """Key for starting timepoint (frame number)."""

    TIME_END = Column.FRAME_STOP_FOR_MODEL
    """Key for ending timepoint (frame number)."""

    TIME_STEP = "frame_step"
    """Key for timepoint (frame number) step size."""

    INCLUDE_TIMEPOINTS = Column.TIMEPOINTS_TO_INCLUDE_FOR_MODEL
    """Key for list of timepoints (frame numbers) to include."""

    Z_START = Column.Z_START_FOR_MODEL
    """Key for starting z-slice index."""

    Z_END = Column.Z_END_FOR_MODEL
    """Key for ending z-slice index."""

    Z_STEP = Column.Z_STEP_FOR_MODEL
    """Key for z-slice index step size."""

    CHANNELS = Column.IMAGE_CHANNELS_TO_LOAD_FOR_MODEL
    """Key for list of channels to load."""

    RESOLUTION = Column.DiffAEData.RESOLUTION
    """Key for resolution level of the image."""

    SCENE = "scene"
    """Key for scene identifier."""

    START_X = Column.DiffAEData.START_X
    """Upper-left x-coordinate of the crop."""

    START_Y = Column.DiffAEData.START_Y
    """Upper-left y-coordinate of the crop."""

    END_X = Column.DiffAEData.END_X
    """Lower-right x-coordinate of the crop."""

    END_Y = Column.DiffAEData.END_Y
    """Lower-right y-coordinate of the crop."""


class CytoDLSaveDataKeys(StrEnum):
    """Metadata keys passed to CytoDL callback object SaveTabularData."""

    FILE_PATH = "filename_or_obj"
    """Key for path to the source image file."""

    TIMEPOINT = "T"
    """Key for timepoint (frame number)."""


NUM_LATENT_FEATURES = 512
"""Number of latent features to extract from DiFFAE model."""

NUM_PCS_TO_ANALYZE = 3
"""Number of top principal components to analyze."""

MAX_PCS_TO_COMPUTE = 100
"""Maximum number of principal components to compute for this project."""

DIFFAE_FEATURE_COLUMN_NAMES = [
    ColumnTemplate.LATENT_FEATURE % i for i in range(NUM_LATENT_FEATURES)
]
"""Full set of column names for original latent features in DiFFAE feature dataframes."""

DIFFAE_PC_COLUMN_NAMES = [ColumnTemplate.PCA_FEATURE % (i + 1) for i in range(NUM_LATENT_FEATURES)]
"""Full set of column names for PCA-transformed features in DiFFAE feature dataframes."""

DIFFAE_PC_COLUMN_NAME_GROUPS: dict[str, list[str]] = {
    "default": DIFFAE_PC_COLUMN_NAMES[:3]
    + [
        Column.DiffAEData.POLAR_ANGLE,
        Column.DiffAEData.POLAR_RADIUS,
        Column.DiffAEData.PC3_FLIPPED,
    ],
    "main_figure": [
        Column.DiffAEData.POLAR_ANGLE,
        Column.DiffAEData.POLAR_RADIUS,
        Column.DiffAEData.PC3_FLIPPED,
    ],
    "supp_figure": [
        Column.DiffAEData.POLAR_ANGLE,
        Column.DiffAEData.POLAR_RADIUS,
        Column.DiffAEData.PC3_FLIPPED,
    ]
    + DIFFAE_PC_COLUMN_NAMES[:10],
    "polar_coord": [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.POLAR_ANGLE],
    "first_3_pcs": DIFFAE_PC_COLUMN_NAMES[:3],
    "first_100_pcs": DIFFAE_PC_COLUMN_NAMES[:100],
    "all": DIFFAE_PC_COLUMN_NAMES,
}

DIFFAE_FEATURE_COLUMN_NAME_GROUPS = {
    "all": DIFFAE_FEATURE_COLUMN_NAMES,
}
