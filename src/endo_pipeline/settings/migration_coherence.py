from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

OPTICAL_FLOW_BASE_FEATURES: tuple[str, ...] = (
    "optical_flow_mean_speed_dt1",
    "optical_flow_mean_unit_vector_dt1",
    "optical_flow_std_speed_dt1",
    "optical_flow_mean_angle_dt1",
    "optical_flow_angle_std_dt1",
    "optical_flow_mean_u_dt1",
    "optical_flow_mean_v_dt1",
    "optical_flow_std_u_dt1",
    "optical_flow_std_v_dt1",
)
"""Base feature names computed per (crop, timepoint, dt) by optical-flow extraction."""

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

MIGRATION_COHERENCE_CROP_PATTERN = "grid"
"""Crop pattern to use for migration coherence analyses."""

DEFAULT_MIGRATION_COHERENCE_FEATURE: str = "optical_flow_mean_unit_vector_dt1"
"""Default optical flow feature to use for migration coherence analyses and plotting."""

MIGRATION_COHERENCE_COLORMAP = "cool"
"""Colormap to use for visualizing migration coherence features."""
