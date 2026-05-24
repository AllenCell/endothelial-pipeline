"""Settings for first passage time analysis."""

from endo_pipeline.settings.column_names import ColumnName as Column

FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME: str = "first_passage_time_statistics"
"""Manifest name for first passage time statistics."""

FIRST_PASSAGE_TIME_PARAMETER_SWEEP_MANIFEST_NAME: str = "first_passage_time_parameter_sweep"
"""Manifest name for first passage time parameter sweep."""

FIRST_PASSAGE_TIME_BIN_SIZES = {
    Column.DiffAEData.POLAR_ANGLE: 15,
    Column.DiffAEData.POLAR_RADIUS: 0.25,
    Column.DiffAEData.PC3_FLIPPED: 0.5,
}
"""Bin sizes for first passage time analysis."""
