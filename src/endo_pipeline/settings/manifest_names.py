"""Manifest name defaults and prefixes."""

from endo_pipeline.settings.literal_types import PatchTypeLiteral

ZARR_IMAGE_MANIFEST_NAME: str = "image_zarr"
"""Name of the Zarr image manifest."""

GRID_BASED_VECTOR_FIELD_MANIFEST_NAME: str = "drift_vector_field_grid_based"
"""Dataframe manifest name for grid-based vector fields."""

VECTOR_FIELD_MANIFEST_NAMES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_VECTOR_FIELD_MANIFEST_NAME,
    "cell_centered": "drift_vector_field_cell_centered",
}
"""Mapping of patch type to vector field dataframe manifest name."""

GRID_BASED_STRATONOVICH_VECTOR_FIELD_MANIFEST_NAME: str = "stratonovich_vector_field_grid_based"
"""Dataframe manifest name for the Stratonovich noise-induced-drift corrected vector field, grid-based."""

STRATONOVICH_VECTOR_FIELD_MANIFEST_NAMES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_STRATONOVICH_VECTOR_FIELD_MANIFEST_NAME,
    "cell_centered": "stratonovich_vector_field_cell_centered",
}
"""Mapping of patch type to Stratonovich-corrected vector field dataframe manifest name."""


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

GRID_BASED_AUTOCORRELATION_MANIFEST_NAME_PREFIX: str = "autocorrelation_grid_based"
"""Dataframe manifest name for autocorrelation of grid-based features."""

AUTOCORRELATION_MANIFEST_NAME_PREFIXES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_AUTOCORRELATION_MANIFEST_NAME_PREFIX,
    "cell_centered": "autocorrelation_cell_centered",
}
"""Mapping of patch type to autocorrelation dataframe manifest name."""

GRID_BASED_NOISE_STATISTICS_MANIFEST_NAME_PREFIX: str = "noise_statistics_grid_based"
"""Dataframe manifest name for noise statistics of grid-based features."""

NOISE_STATISTICS_MANIFEST_NAME_PREFIXES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_NOISE_STATISTICS_MANIFEST_NAME_PREFIX,
    "cell_centered": "noise_statistics_cell_centered",
}
"""Mapping of patch type to noise statistics dataframe manifest name."""

GRID_BASED_NOISE_CORRELATIONS_MANIFEST_NAME_PREFIX: str = "noise_correlation_functions_grid_based"
"""Extra manifest name for full two-timepoint noise correlation functions of grid-based features."""

NOISE_CORRELATIONS_MANIFEST_NAME_PREFIXES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_NOISE_CORRELATIONS_MANIFEST_NAME_PREFIX,
    "cell_centered": "noise_correlation_functions_cell_centered",
}
"""Mapping of patch type to two-timepoint noise correlation function extra manifest name."""

GRID_BASED_NOISE_PERIODOGRAM_MANIFEST_NAME_PREFIX: str = "noise_periodograms_grid_based"
"""Extra manifest name for noise residual periodograms of grid-based features."""

NOISE_PERIODOGRAM_MANIFEST_NAME_PREFIXES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_NOISE_PERIODOGRAM_MANIFEST_NAME_PREFIX,
    "cell_centered": "noise_periodograms_cell_centered",
}
"""Mapping of patch type to noise residual periodogram extra manifest name."""

GRID_BASED_NOISE_AMPLITUDE_MANIFEST_NAME_PREFIX: str = "noise_amplitude_grid_based"
"""Dataframe manifest name for the histogram-based noise amplitude tensor field of grid-based features."""

NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES: dict[PatchTypeLiteral, str] = {
    "grid_based": GRID_BASED_NOISE_AMPLITUDE_MANIFEST_NAME_PREFIX,
    "cell_centered": "noise_amplitude_cell_centered",
}
"""Mapping of patch type to noise amplitude tensor field dataframe manifest name."""
