"""Settings for coefficient of variation analysis."""

from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

COV_VS_TIME_YLIM_DICT: dict[str, tuple[float, float]] = {
    ColumnName.POLAR_ANGLE.value: (0.0, 1.5),
    ColumnName.POLAR_RADIUS.value: (0.0, 1.5),
    ColumnName.PC3_FLIPPED.value: (0.0, 1.5),
}
"""Bin limits for coefficient of variation vs time plots in visualization."""
