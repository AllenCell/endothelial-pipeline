"""Defaults for visualizing histogram of individual features over time."""

from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

FEATURES_FOR_HISTOGRAM_VIS: tuple[str, ...] = (
    ColumnName.POLAR_ANGLE,
    ColumnName.POLAR_RADIUS,
    ColumnName.PC3_FLIPPED,
)
"""Default features to visualize in 1d histogram-over-time plots."""

BIN_WIDTHS_FOR_HISTOGRAMS: tuple[float, ...] = (0.05, 0.05, 0.05)
"""Default bin widths for histogram-over-time plots for polar angle, polar radius, and PC3 values."""

BIN_LIMITS_RHO: tuple[float, float] = (-1.5, 2.5)
"""Default range for -1 x PC3 (ColumnName.PC3_FLIPPED) values in histogram-over-time plots."""
