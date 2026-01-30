"""Default settings for polar coordinate analysis and visualization."""

from numpy import pi

from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

BIN_WIDTHS_POLAR: tuple[float, float] = (0.05, 0.05)
"""Bin widths for polar coordinate density and flow field estimation in the order [angle, radius]."""

DEFAULT_DATASET_COLLECTION_POLAR_VIS: str = "diffae_model_training"
"""Default dataset collection for polar coordinate visualization workflow."""

POLAR_COLUMN_NAMES = [ColumnName.POLAR_ANGLE, ColumnName.POLAR_RADIUS]
"""Column names for polar coordinates in the DiffAE feature dataframe, in the order [angle, radius]."""

BIN_LIMITS_POLAR: list[tuple[float, float]] = [(-pi, pi), (0.0, 3.5)]
"""Bin limits for polar coordinate analysis, in the order [angle, radius]."""

RESCALE_THETA: bool = True
"""Whether to rescale polar angle coordinate to [0, pi] range for analysis and visualization."""

BIN_LIMITS_THETA_RESCALED: tuple[float, float] = (0.0, pi)
"""Bin limits for rescaled polar angle coordinate analysis and visualization."""

THETA_RESCALED_PERIOD: float = pi
"""Period for rescaled polar angle coordinate."""

TICK_STEP_NUM: int = 7
"""Number of axes ticks for coordinate axis in histogram plots."""

BEHAVES_LIKE_MIN_SHEAR_STRESS: tuple[str, ...] = (
    "20250428_20X",
    "20250604_20X",
    "20250716_20X",
)
"""Datasets with distribution of polar angle coordinate qualitatively similar to MIN shear stress datasets."""
