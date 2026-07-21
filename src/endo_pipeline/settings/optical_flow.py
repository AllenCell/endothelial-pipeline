"""Optical-flow workflow constants and defaults."""

from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.literal_types import PatchTypeLiteral

DEMO_MAX_TRACKED_CROPS_TO_PLOT: int = 2
"""Maximum number of tracked crops to plot in the coherence time series diagnostic."""

QUIVER_GRID_DIVISIONS: int = 8
"""Number of grid divisions along each axis for quiver plot sub-sampling.

The crop is divided into an ``N x N`` grid and flow vectors are averaged
within each cell, producing one arrow per cell in the diagnostic quiver
overlay.
"""

DEFAULT_OPTICAL_FLOW_COLLECTION: str = "diffae_model_training"
"""Default dataset collection for the optical-flow feature workflow."""

DIFFAE_DATAFRAME_METADATA_TO_COMPUTE: tuple[str, ...] = (
    ColumnName.DATASET,
    ColumnName.POSITION,
    ColumnName.TIMEPOINT,
    ColumnName.CROP_INDEX,
    ColumnName.DiffAEData.START_X,
    ColumnName.DiffAEData.START_Y,
    ColumnName.DiffAEData.END_X,
    ColumnName.DiffAEData.END_Y,
)
"""Metadata columns from Diff AE feature dataframe needed for optical flow computations."""

OPTICAL_FLOW_COLUMNS_TO_COMPUTE: dict[PatchTypeLiteral, tuple[str, ...]] = {
    "grid_based": DIFFAE_DATAFRAME_METADATA_TO_COMPUTE,
    "cell_centered": DIFFAE_DATAFRAME_METADATA_TO_COMPUTE
    + (
        ColumnName.TRACK_ID,
        ColumnName.DiffAEData.CROP_SIZE_X,
    ),
}
"""Metadata columns keyed by patch type."""

DEFAULT_EMA_ALPHA: float = 0.1
"""Default EMA smoothing alpha value for temporal coherence smoothing."""

OPTICAL_FLOW_ATTACHMENT: float = 2.5  # z-score normalisation compresses dynamic range
"""TVL1 attachment (lambda)."""

DEFAULT_OPTICAL_FLOW_MAX_DT: int = 1
"""Maximum frame gap for multi-scale optical-flow sweep (dt = 1 ... MAX_DT)."""
