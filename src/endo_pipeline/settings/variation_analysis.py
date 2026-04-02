"""Settings for coefficient of variation analysis."""

from endo_pipeline.settings.column_names import ColumnName as Column

COV_VS_TIME_YLIM_DICT: dict[str, tuple[float, float]] = {
    Column.DiffAEData.POLAR_ANGLE.value: (0.0, 1.5),
    Column.DiffAEData.POLAR_RADIUS.value: (0.0, 1.5),
    Column.DiffAEData.PC3_FLIPPED.value: (0.0, 1.5),
}
"""Bin limits for coefficient of variation vs time plots in visualization."""

DEFAULT_COV_ANALYSIS_COLUMNS: tuple[str, ...] = (
    Column.DiffAEData.POLAR_ANGLE.value,
    Column.DiffAEData.POLAR_RADIUS.value,
    Column.DiffAEData.PC3_FLIPPED.value,
)
"""Default column names to include in coefficient of variation analysis."""

TIME_WINDOW_BIN_SIZE: int = 12
"""Default size of rolling windows over time for binned per-crop variance."""
