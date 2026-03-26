"""Default settings for polar coordinate analysis and visualization."""

from numpy import pi

from endo_pipeline.settings.column_names import ColumnName as Column

METADATA_COLUMNS_TO_KEEP: tuple[str | Column.DiffAEData, ...] = (
    Column.DATASET,
    Column.POSITION,
    Column.TIMEPOINT,
    Column.CROP_INDEX,
    Column.DiffAEData.START_X,
    Column.DiffAEData.START_Y,
    Column.DiffAEData.END_X,
    Column.DiffAEData.END_Y,
)

TRACK_METADATA_COLUMNS_TO_KEEP: tuple[str | Column.SegDataFilters, ...] = (
    Column.TRACK_ID,
    Column.TRACK_LENGTH,
    Column.SegDataFilters.IS_INCLUDED,
)

DYNAMICS_COLUMN_NAMES: tuple[Column.DiffAEData, ...] = (
    Column.DiffAEData.POLAR_ANGLE,
    Column.DiffAEData.POLAR_RADIUS,
    Column.DiffAEData.PC3_FLIPPED,
)
"""Column names in the DiffAE feature dataframe to use for dynamics analysis and
visualization."""

BIN_WIDTHS_DYNAMICS: dict[Column.DiffAEData, float] = {
    Column.DiffAEData.POLAR_ANGLE: 0.05,
    Column.DiffAEData.POLAR_RADIUS: 0.05,
    Column.DiffAEData.PC3_FLIPPED: 0.05,
}
"""Bin widths for each coordinate in dynamics analysis and visualization."""

BIN_LIMITS_DYNAMICS: dict[Column.DiffAEData, tuple[float, float]] = {
    Column.DiffAEData.POLAR_ANGLE: (-pi, pi),
    Column.DiffAEData.POLAR_RADIUS: (0.0, 3.5),
    Column.DiffAEData.PC3_FLIPPED: (-1.5, 2.5),
}
"""Bin limits for each coordinate in dynamics analysis and visualization."""

DEFAULT_DATASETS_DYNAMICS_VIS: str = "diffae_model_training"
"""Default dataset collection for dynamics visualization workflows."""

RESCALE_THETA: bool = True
"""Whether to rescale polar angle coordinate to [0, pi] range for analysis and
visualization."""

BIN_LIMITS_THETA_RESCALED: tuple[float, float] = (0.0, pi)
"""Bin limits for rescaled polar angle coordinate analysis and visualization."""

PERIOD_THETA_RESCALED: float = pi
"""Period for rescaled polar angle coordinate."""

KERNEL_NAMES_DYNAMICS: dict[Column.DiffAEData, str] = {
    Column.DiffAEData.POLAR_ANGLE: "periodic",
    Column.DiffAEData.POLAR_RADIUS: "gaussian",
    Column.DiffAEData.PC3_FLIPPED: "gaussian",
}
"""Kernel names for each coordinate in dynamics analysis and visualization."""

KERNEL_BANDWIDTHS_DYNAMICS: dict[Column.DiffAEData, float] = {
    Column.DiffAEData.POLAR_ANGLE: 0.15,
    Column.DiffAEData.POLAR_RADIUS: 0.15,
    Column.DiffAEData.PC3_FLIPPED: 0.15,
}

BIN_LIMIT_PERCENTILE_CUTOFF: float = 2.5
"""Percentile cutoff for getting bin limits for computing Kramer-Moyal
coefficients."""

HISTOGRAM_THRESHOLD_FOR_MASKING: float = 0.05
"""Histogram threshold for masking in dynamics visualization workflows."""

MAX_MSD_LAG: int = 24
"""Maximum time lag (in number of frames) to consider for mean squared
displacement calculation."""

MSD_Y_AXIS_LIMITS: tuple[float, float] = (2e-3, 1e0)
"""Axes limits for mean squared displacement plots."""

MINIMUM_MSD_TRACK_LENGTH: int = 150
"""Minimum track length (in number of timepoints) to include in mean squared
displacement calculation."""
