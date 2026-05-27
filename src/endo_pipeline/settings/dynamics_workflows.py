"""Default settings for polar coordinate analysis and visualization."""

from math import pi

from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KernelName
from endo_pipeline.settings.column_names import ColumnName as Column

METADATA_COLUMNS_TO_KEEP: dict[str, tuple[str | Column.DiffAEData, ...]] = {
    "grid": (
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.CROP_INDEX,
        Column.DiffAEData.START_X,
        Column.DiffAEData.START_Y,
        Column.DiffAEData.END_X,
        Column.DiffAEData.END_Y,
    ),
    "tracked": (
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.CROP_INDEX,
        Column.DiffAEData.START_X,
        Column.DiffAEData.START_Y,
        Column.DiffAEData.END_X,
        Column.DiffAEData.END_Y,
        Column.TRACK_ID,
        Column.TRACK_LENGTH,
    ),
}

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
    Column.DiffAEData.POLAR_ANGLE: (0.0, pi),
    Column.DiffAEData.POLAR_RADIUS: (0.0, 3.5),
    Column.DiffAEData.PC3_FLIPPED: (-1.5, 2.5),
}
"""Bin limits for each coordinate in dynamics analysis and visualization."""

DEFAULT_DATASETS_DYNAMICS_VIS: str = "diffae_model_training"
"""Default dataset collection for dynamics visualization workflows."""

TIME_STEP_IN_MINUTES: int = 5
"""Time step in minutes between consecutive time points for flow field estimation."""

TIME_STEP_IN_HOURS: float = TIME_STEP_IN_MINUTES / 60
"""Time step in hours between consecutive time points for flow field estimation."""

RESCALE_THETA: bool = True
"""Whether to rescale polar angle coordinate to [0, pi] range for analysis and
visualization."""

POLAR_ANGLE_RANGE: tuple[float, float] = (0.0, pi)
"""Range of polar angle coordinate (dependent on RESCALE_THETA) for analysis and visualization."""

POLAR_ANGLE_PERIOD: float = pi
"""Period for polar angle coordinate (dependent on RESCALE_THETA)."""

RESCALED_THETA_PERIOD: float = POLAR_ANGLE_PERIOD + pi * (1 - RESCALE_THETA)
"""Rescaled period if polar angle is rescaled."""

KERNEL_NAMES_DYNAMICS: dict[Column.DiffAEData, KernelName] = {
    Column.DiffAEData.POLAR_ANGLE: "periodic",
    Column.DiffAEData.POLAR_RADIUS: "gaussian",
    Column.DiffAEData.PC3_FLIPPED: "gaussian",
}
"""Map of column name to kernel names."""

KERNEL_BANDWIDTHS_DYNAMICS: dict[Column.DiffAEData, float] = {
    Column.DiffAEData.POLAR_ANGLE: 0.15,
    Column.DiffAEData.POLAR_RADIUS: 0.15,
    Column.DiffAEData.PC3_FLIPPED: 0.15,
}
"""Map of column name to kernel bandwidths."""

KERNEL_PERIODS_DYNAMICS: dict[Column.DiffAEData, float | None] = {
    Column.DiffAEData.POLAR_ANGLE: RESCALED_THETA_PERIOD,
    Column.DiffAEData.POLAR_RADIUS: None,
    Column.DiffAEData.PC3_FLIPPED: None,
}
"""Map of column name to kernel periods."""


BIN_LIMIT_PERCENTILE_CUTOFF: float = 2.5
"""Percentile cutoff for getting bin limits for computing Kramer-Moyal
coefficients."""

NUM_INIT_SAMPLES: int = 250
"""Number of sampled initial points for root finding in 3D flow field analysis."""

SAMPLER_RANDOM_SEED: int = 47
"""Random seed for initial point sampling in 3D flow field analysis."""

UPPER_PERCENTILE_FOR_FILTERING_FPTS: float = 95.0
"""Upper percentile threshold for stable fixed point identification in 3D flow field analysis."""

LOWER_PERCENTILE_FOR_FILTERING_FPTS: float = 5.0
"""Lower percentile threshold for stable fixed point identification in 3D flow field analysis."""

LONG_TRACK_THRESHOLD_LENGTH: int = 72
"""
Minimum track length (in number of timepoints) to include in analyses of
long-timescale statistics (e.g., mean squared displacement) in dynamics
workflows.
"""
