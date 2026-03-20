"""Optical-flow workflow constants and defaults.

All tuneable constants for the optical-flow feature-extraction pipeline
live here so that they can be adjusted in one place without touching
compute, I/O, or visualisation code.
"""

# ---------------------------------------------------------------------------
# Multi-scale coherence
# ---------------------------------------------------------------------------
COHERENCE_BOX_SIZES: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
)
"""Non-overlapping box sizes (in pixels) for multi-scale coherence."""

# ---------------------------------------------------------------------------
# Demo / scan-mode diagnostic limits
# ---------------------------------------------------------------------------
DEMO_SCAN_N_CROPS: int = 6
"""Number of crops to visualize in demo/scan mode diagnostic plots."""

DEMO_SCAN_N_PAIRS: int = 10
"""Number of frame pairs to visualize in demo/scan mode diagnostic plots."""

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
"""Default dataframe manifest name for optical-flow features."""

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
OPTICAL_FLOW_BASE_FEATURES: list[str] = [
    "optical_flow_mean_speed",
    "optical_flow_mean_unit_vector",
    "optical_flow_std_speed",
    "optical_flow_mean_angle",
    "optical_flow_angle_std",
    "optical_flow_mean_u",
    "optical_flow_mean_v",
    "optical_flow_std_u",
    "optical_flow_std_v",
]
"""Base feature names computed per (crop, timepoint, dt) by optical-flow extraction."""

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
