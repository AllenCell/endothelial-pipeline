"""Default settings for polar coordinate analysis and visualization."""

from numpy import pi

from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

METADATA_COLUMNS_TO_KEEP: tuple[str, ...] = (
    ColumnName.DATASET.value,
    ColumnName.POSITION.value,
    ColumnName.TIMEPOINT.value,
    ColumnName.START_X.value,
    ColumnName.START_Y.value,
    ColumnName.END_X.value,
    ColumnName.END_Y.value,
)

DYNAMICS_COLUMN_NAMES: tuple[str, ...] = (
    ColumnName.POLAR_ANGLE.value,
    ColumnName.POLAR_RADIUS.value,
    ColumnName.PC3_FLIPPED.value,
)
"""Column names in the DiffAE feature dataframe to use for dynamics analysis and
visualization."""

BIN_WIDTHS_DYNAMICS: dict[str, float] = {
    ColumnName.POLAR_ANGLE.value: 0.05,
    ColumnName.POLAR_RADIUS.value: 0.05,
    ColumnName.PC3_FLIPPED.value: 0.05,
}
"""Bin widths for each coordinate in dynamics analysis and visualization."""

BIN_LIMITS_DYNAMICS: dict[str, tuple[float, float]] = {
    ColumnName.POLAR_ANGLE.value: (-pi, pi),
    ColumnName.POLAR_RADIUS.value: (0.0, 3.5),
    ColumnName.PC3_FLIPPED.value: (-1.5, 2.5),
}
"""Bin limits for each coordinate in dynamics analysis and visualization."""

DEFAULT_DATASET_DYNAMICS_VIS: str = "20250618_20X"
"""Default dataset for dynamics visualization workflows."""

RESCALE_THETA: bool = True
"""Whether to rescale polar angle coordinate to [0, pi] range for analysis and
visualization."""

BIN_LIMITS_THETA_RESCALED: tuple[float, float] = (0.0, pi)
"""Bin limits for rescaled polar angle coordinate analysis and visualization."""

PERIOD_THETA_RESCALED: float = pi
"""Period for rescaled polar angle coordinate."""

KERNEL_NAMES_DYNAMICS: dict[str, str] = {
    ColumnName.POLAR_ANGLE.value: "periodic",
    ColumnName.POLAR_RADIUS.value: "gaussian",
    ColumnName.PC3_FLIPPED.value: "gaussian",
}
"""Kernel names for each coordinate in dynamics analysis and visualization."""

KERNEL_BANDWIDTHS_DYNAMICS: dict[str, float] = {
    ColumnName.POLAR_ANGLE.value: 0.15,
    ColumnName.POLAR_RADIUS.value: 0.15,
    ColumnName.PC3_FLIPPED.value: 0.15,
}

BIN_LIMIT_PERCENTILE_CUTOFF: float = 2.5
"""Percentile cutoff for getting bin limits for computing Kramer-Moyal
coefficients."""

NUM_PCS_TO_FIT_FOR_DYNAMICS: int = 3
"""Number of principal components to fit for dynamics analysis and
visualization."""

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
