from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.literal_types import PatchTypeLiteral

OPTICAL_FLOW_DATAFRAME_MERGE_COLUMNS: tuple[str | ColumnName.DiffAEData, ...] = (
    ColumnName.DATASET,
    ColumnName.POSITION,
    ColumnName.TIMEPOINT,
    ColumnName.DiffAEData.START_X,
    ColumnName.DiffAEData.START_Y,
)
"""Column names to merge on when adding optical flow features to another dataframe."""

MIGRATION_COHERENCE_PATCH_TYPE: PatchTypeLiteral = "grid_based"
"""Patch type to use for migration coherence analyses."""

MIGRATION_COHERENCE_HIST_PLOT_KDE: bool = True
"""Whether to plot a kernel density estimate on migration coherence histogram plots."""

MIGRATION_COHERENCE_COLORMAP: str = "cool"
"""Colormap to use for visualizing migration coherence features."""

MIGRATION_COHERENCE_COLORMAP_BIN_SIZE: float = 0.25
"""Bin size for binned mean colormap in migration coherence analyses."""
