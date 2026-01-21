"""Default settings for polar coordinate analysis and visualization."""

from numpy import pi

from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

KERNEL_BANDWIDTH_POLAR: float = 0.175
"""Kernel bandwidth for 1D polar coordinate density and flow field estimation."""

BIN_WIDTH_POLAR: float = 0.05
"""Bin width for 1D polar coordinate density and flow field estimation."""

DEFAULT_DATASET_COLLECTION_POLAR_VIS: str = "diffae_model_training"
"""Default dataset collection for polar coordinate visualization workflow."""

POLAR_COLUMN_NAMES = [ColumnName.POLAR_ANGLE, ColumnName.POLAR_RADIUS]

BIN_LIMITS_ANGLE: tuple[float, float] = (-pi, pi)
"""Bin limits for the angular coordinate in polar coordinate analysis."""

BIN_LIMITS_RADIUS: tuple[float, float] = (0, 3.5)
"""Bin limits for the radial coordinate in polar coordinate analysis."""

TICK_STEP_NUM: int = 7
"""Number of axes ticks for coordinate axis in histogram plots."""
