from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

OPTICAL_FLOW_DATFRAME_MANIFEST_NAME: str = "optical_flow_bf"
"""Name of dataframe manifest containing optical flow features."""

OPTICAL_FLOW_DATAFRAME_MERGE_COLUMNS: tuple[str, ...] = (
    ColumnName.DATASET,
    ColumnName.POSITION,
    ColumnName.TIMEPOINT,
    ColumnName.START_X,
    ColumnName.START_Y,
)
"""Column names to merge on when adding optical flow features to another dataframe."""

MIGRATION_COHERENCE_CROP_PATTERN: str = "grid"
"""Crop pattern to use for migration coherence analyses."""

DEFAULT_MIGRATION_COHERENCE_FEATURE: str = "optical_flow_mean_unit_vector_dt1"
"""Default optical flow feature to use for migration coherence analyses and plotting."""

MIGRATION_COHERENCE_HIST_FIGSIZE: tuple[float, float] = (4, 2.5)
"""Figure size (width, height) in inches for migration coherence histogram plots."""

MIGRATION_COHERENCE_HIST_NUM_BINS: int = 50
"""Number of bins to use for migration coherence histogram plots."""

MIGRATION_COHERENCE_HIST_BINWIDTH: float = 0.02
"""Width of each bin for migration coherence histogram plots."""

MIGRATION_COHERENCE_HIST_PLOT_KDE: bool = True
"""Whether to plot a kernel density estimate on migration coherence histogram plots."""

MIGRATION_COHERENCE_COLORMAP: str = "cool"
"""Colormap to use for visualizing migration coherence features."""

MIGRATION_COHERENCE_COLORMAP_BIN_SIZE: float = 0.25
"""Bin size for binned mean colormap in migration coherence analyses."""
