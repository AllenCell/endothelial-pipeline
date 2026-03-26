"""Global settings for workflow comparing feature densities between cell-centric and grid-based crops."""

from endo_pipeline.settings.column_names import ColumnName as Column

DENSITY_PLOT_FEATURES: tuple[str, ...] = (
    Column.DiffAEData.POLAR_ANGLE,
    Column.DiffAEData.POLAR_RADIUS,
    Column.DiffAEData.PC3_FLIPPED,
)
"""Column names of features to compare densities for."""

DENSITY_PLOT_METADATA_COLUMNS_TO_COMPUTE: tuple[str, ...] = (
    Column.DATASET,
    Column.POSITION,
    Column.TIMEPOINT,
)
"""Column names of metadata to include when filtering dataframes for density comparison plots."""

DENSITY_PLOT_DEFAULT_DATASET: str = "20250818_20X"
"""Default dataset to use for density comparison plots."""

DENSITY_PLOT_KDE_BANDWIDTH: float = 0.2
"""Default bandwidth for KDE density plots."""

DENSITY_PLOT_KWARGS_GRID_CROPS: dict = {
    "label": "grid-based crops",
    "color": "k",
    "linewidth": 2.75,
    "linestyle": "--",
}
"""Default plotting keyword arguments for density plots of grid-based crops."""

DENSITY_PLOT_KWARGS_TRACKED_CROPS: dict = {
    "label": "cell-centric crops",
    "color": "k",
    "linewidth": 2.75,
    "linestyle": "-",
}
"""Default plotting keyword arguments for density plots of tracked crops."""

SAVE_FIG_FILE_FORMATS: tuple[str, ...] = (".png", ".pdf")
"""File formats to save figures in."""
