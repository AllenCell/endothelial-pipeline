OPTICAL_FLOW_BASE_FEATURES: list = [
    "optical_flow_mean_speed_dt1",
    "optical_flow_mean_unit_vector_dt1",
    "optical_flow_std_speed_dt1",
    "optical_flow_mean_angle_dt1",
    "optical_flow_angle_std_dt1",
    "optical_flow_mean_u_dt1",
    "optical_flow_mean_v_dt1",
    "optical_flow_std_u_dt1",
    "optical_flow_std_v_dt1",
]
"""Base feature names computed per (crop, timepoint, dt) by optical-flow extraction."""

OPTICAL_FLOW_DATFRAME_MANIFEST_NAME = "optical_flow_bf"
"""Name of dataframe manifest containing optical flow features."""

MINIMUM_TRACK_LENGTH_FOR_MIGRATION_COHERENCE = 200
"""Minimum track length (number of timepoints) for inclusion in migration coherence analyses."""

DEFAULT_MIGRATION_COHERENCE_FEATURE: str = "optical_flow_mean_unit_vector_dt1"
"""Default optical flow feature to use for migration coherence analyses and plotting."""

MIGRATION_COHERENCE_COLORMAP = "cool"
"""Colormap to use for visualizing migration coherence features."""
