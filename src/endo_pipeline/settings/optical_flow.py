"""Optical-flow workflow constants and defaults.

All tuneable constants for the optical-flow feature-extraction pipeline
live here so that they can be adjusted in one place without touching
compute, I/O, or visualisation code.
"""

from endo_pipeline.settings.column_names import ColumnName

# ---------------------------------------------------------------------------
# Demo / scan-mode diagnostic limits
# ---------------------------------------------------------------------------
DEMO_SCAN_N_CROPS: int = 6
"""Number of crops to visualize in demo/scan mode diagnostic plots."""

DEMO_SCAN_N_PAIRS: int = 10
"""Number of frame pairs to visualize in demo/scan mode diagnostic plots."""

DEMO_MAX_TRACKED_CROPS_TO_PLOT: int = 2
"""Maximum number of tracked crops to plot in the coherence time series diagnostic."""

# ---------------------------------------------------------------------------
# Quiver plot
# ---------------------------------------------------------------------------
QUIVER_GRID_DIVISIONS: int = 8
"""Number of grid divisions along each axis for quiver plot sub-sampling.

The crop is divided into an ``N x N`` grid and flow vectors are averaged
within each cell, producing one arrow per cell in the diagnostic quiver
overlay.
"""

# ---------------------------------------------------------------------------
# Dataset / manifest defaults
# ---------------------------------------------------------------------------
DEFAULT_OPTICAL_FLOW_COLLECTION: str = "diffae_model_training"
"""Default dataset collection for the optical-flow feature workflow."""

DEFAULT_OPTICAL_FLOW_MANIFEST_NAME: str = "optical_flow_bf"
"""Default dataframe manifest name (prefix) for optical-flow features."""

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

OPTICAL_FLOW_COLUMNS_TO_COMPUTE: dict[str, tuple[str, ...]] = {
    "grid": DIFFAE_DATAFRAME_METADATA_TO_COMPUTE,
    "tracked": DIFFAE_DATAFRAME_METADATA_TO_COMPUTE
    + (
        ColumnName.TRACK_ID,
        ColumnName.DiffAEData.CROP_SIZE_X,
    ),
}
"""Metadata columns keyed by crop pattern (``'grid'`` or ``'tracked'``)."""

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
OPTICAL_FLOW_COMPUTE_FEATURES: list[str] = [
    ColumnName.OpticalFlow.SPEED_MEAN_BASE,
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN_BASE,
    ColumnName.OpticalFlow.SPEED_STD_BASE,
    ColumnName.OpticalFlow.ANGLE_MEAN_BASE,
    ColumnName.OpticalFlow.ANGLE_STD_BASE,
    ColumnName.OpticalFlow.U_MEAN_BASE,
    ColumnName.OpticalFlow.V_MEAN_BASE,
    ColumnName.OpticalFlow.U_STD_BASE,
    ColumnName.OpticalFlow.V_STD_BASE,
]
"""Core features always computed by :func:`compute_flow_statistics`."""

OPTICAL_FLOW_FAST_FEATURES: list[str] = [
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN_FAST_BASE,
    ColumnName.OpticalFlow.SPEED_ABOVE_1_COUNT_BASE,
]
"""Features gated on ``--compute-fast-coherence`` (speed > threshold)."""

OPTICAL_FLOW_RADIAL_FEATURES: list[str] = [
    ColumnName.OpticalFlow.RADIAL_COHERENCE_BASE,
    ColumnName.OpticalFlow.RADIAL_COHERENCE_WEIGHTED_BASE,
]
"""Features gated on ``--compute-radial-coherence``."""

# Backward-compatible union used by compute.py NaN-fallback when all
# optional features are enabled.
OPTICAL_FLOW_BASE_FEATURES: list[str] = (
    OPTICAL_FLOW_COMPUTE_FEATURES + OPTICAL_FLOW_FAST_FEATURES + OPTICAL_FLOW_RADIAL_FEATURES
)
"""Union of all per-(crop, timepoint, dt) features returned by
:func:`compute_flow_statistics` when every optional group is enabled."""

# EMA coherence feature stems (the alpha tag and _dt suffix are added
# dynamically in the workflow and in :func:`build_optical_flow_feature_cols`).
OPTICAL_FLOW_EMA_STEMS: list[str] = [
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN_BASE,
]
"""Base coherence feature names that always receive EMA smoothing."""

OPTICAL_FLOW_EMA_FAST_STEMS: list[str] = [
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN_FAST_BASE,
]
"""Coherence feature names that receive EMA smoothing only when fast coherence is enabled."""

OPTICAL_FLOW_EMA_RADIAL_STEMS: list[str] = [
    ColumnName.OpticalFlow.RADIAL_COHERENCE_BASE,
    ColumnName.OpticalFlow.RADIAL_COHERENCE_WEIGHTED_BASE,
]
"""Coherence feature names that receive EMA smoothing only when radial coherence is enabled."""

# ---------------------------------------------------------------------------
# Default CLI flag values
# ---------------------------------------------------------------------------
DEFAULT_EMA_ALPHAS: tuple[float, ...] = (0.1,)
"""Default EMA smoothing alpha values for temporal coherence smoothing."""

DEFAULT_SPEED_THRESHOLD: float = 1.0
"""Default speed threshold for fast-coherence feature computation."""

# ---------------------------------------------------------------------------
# Channel-aware parameters
# ---------------------------------------------------------------------------
OPTICAL_FLOW_CHANNEL_PERCENTILE: dict[str, int] = {
    "EGFP": 95,  # sparse fluorescence → exclude empty background
    "BF": 0,  # dense texture → keep all pixels
}
"""Intensity percentile thresholds per channel for optical-flow masking."""

OPTICAL_FLOW_CHANNEL_ATTACHMENT: dict[str, float] = {
    "EGFP": 7.5,  # half of skimage default (15); wider normalised range
    "BF": 2.5,  # z-score normalisation compresses dynamic range
}
"""TVL1 attachment (lambda) per channel.

Smaller values produce smoother flow fields.  BF uses a lower value
because z-score normalisation compresses the intensity range, so a
weaker data-fidelity term avoids fitting noise."""

# ---------------------------------------------------------------------------
# Temporal sweep
# ---------------------------------------------------------------------------
DEFAULT_OPTICAL_FLOW_MAX_DT: int = 1
"""Maximum frame gap for multi-scale optical-flow sweep (dt = 1 ... MAX_DT)."""

# ---------------------------------------------------------------------------
# Thread pinning
# ---------------------------------------------------------------------------
DEFAULT_OMP_NUM_THREADS: str = "1"
"""Default OMP_NUM_THREADS for optical-flow workers (pinned to avoid over-subscription)."""

DEFAULT_OPENBLAS_NUM_THREADS: str = "1"
"""Default OPENBLAS_NUM_THREADS for optical-flow workers."""

NUM_IO_WORKERS: int = 16
"""Concurrent I/O workers for dask compute and ThreadPoolExecutor.

Determined empirically on compute nodes (512 GB RAM, 128 physical /
256 logical CPU cores, 4x A100 80 GB GPUs).  TVL1 is pinned to one
thread per call (OMP_NUM_THREADS=1), so the bottleneck is NFS read
throughput rather than CPU.  At 16 concurrent workers, each holding
~0.5 GB per frame, peak memory is ~8 GB — well within budget — while
NFS throughput is fully saturated; beyond 16 workers wall-clock time
plateaus but memory grows linearly with no additional speedup.
"""
