"""Default settings for polar coordinate analysis and visualization."""

from numpy import pi

from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

KERNEL_BANDWIDTH_POLAR: float = 0.175
"""Kernel bandwidth for 1D polar coordinate density and flow field estimation."""

SPLIT_BY_FLOW: bool = True
"""Whether to split data by flow conditions for polar coordinate analysis."""

BIN_WIDTHS_POLAR: tuple[float, float] = (0.05, 0.05)
"""Bin widths for polar coordinate density and flow field estimation in the order [angle, radius]."""

DEFAULT_DATASET_COLLECTION_POLAR_VIS: str = "diffae_model_training"
"""Default dataset collection for polar coordinate visualization workflow."""

POLAR_COLUMN_NAMES = [ColumnName.POLAR_ANGLE, ColumnName.POLAR_RADIUS]
"""Column names for polar coordinates in the DiffAE feature dataframe, in the order [angle, radius]."""

BIN_LIMITS_POLAR = [(-pi, pi), (0.0, 3.5)]
"""Bin limits for polar coordinate analysis, in the order [angle, radius]."""

TICK_STEP_NUM: int = 7
"""Number of axes ticks for coordinate axis in histogram plots."""
