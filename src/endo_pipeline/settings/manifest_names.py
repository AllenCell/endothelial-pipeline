"""Manifest name defaults and prefixes."""

from endo_pipeline.settings.literal_types import PatchTypeLiteral

ZARR_IMAGE_MANIFEST_NAME: str = "image_zarr"
"""Name of the Zarr image manifest."""

DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD: str = "drift_vector_field"
"""Prefix for vector field dataframe manifest name."""

DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS: str = "drift_fixed_points"
"""Prefix for fixed points dataframe manifest name."""

GRID_BASED_FIXED_POINT_MANIFEST_NAME: str = "drift_fixed_points_grid_based"
"""Dataframe manifest name for grid-based fixed points."""

FIXED_POINT_MANIFEST_NAMES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_FIXED_POINT_MANIFEST_NAME,
    "cell_centered": "drift_fixed_points_cell_centered",
}
"""Mapping of patch type to fixed points dataframe manifest name."""

GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME: str = "bootstrapped_fixed_points_grid_based"
"""Dataframe manifest name for grid-based bootstrapping results."""

BOOTSTRAPPING_MANIFEST_NAMES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME,
    "cell_centered": "bootstrapped_fixed_points_cell_centered",
}
"""Mapping of patch type to bootstrapping dataframe manifest name."""

GRID_BASED_OPTICAL_FLOW_MANIFEST_NAME = "optical_flow_bf_grid_based"
"""Dataframe manifest name for grid-based optical flow features."""

CELL_CENTERED_OPTICAL_FLOW_MANIFEST_NAME = "optical_flow_bf_cell_centered"
"""Dataframe manifest name for cell-centered optical flow features."""

OPTICAL_FLOW_MANIFEST_NAMES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_OPTICAL_FLOW_MANIFEST_NAME,
    "cell_centered": CELL_CENTERED_OPTICAL_FLOW_MANIFEST_NAME,
}
"""Mapping of patch type to optical flow dataframe manifest name."""

GRID_BASED_AUTOCORRELATION_MANIFEST_NAME: str = "autocorrelation_grid_based"
"""Dataframe manifest name for autocorrelation of grid-based features."""

AUTOCORRELATION_MANIFEST_NAMES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_AUTOCORRELATION_MANIFEST_NAME,
    "cell_centered": "autocorrelation_cell_centered",
}
"""Mapping of patch type to autocorrelation dataframe manifest name."""
