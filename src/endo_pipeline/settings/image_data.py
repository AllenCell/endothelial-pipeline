"""Global settings for image data processing in the endo_pipeline."""

LOWER_Z_SLICE_OFFSET = 4
"""How many slices below the "center" Z-plane to include in projections."""

UPPER_Z_SLICE_OFFSET = 11
"""How many slices above the "center" Z-plane to include in projections."""

Z_SLICE_OFFSETS = (LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET)
"""Tuple containing the lower and upper Z-offsets for projections."""

LOG_EPSILON = 1e-12
"""Small constant used to avoid log(0) errors."""

NUM_ZSLICES = 25
"""Number of z-slices per timepoint."""

IF_INTEGRATION_SAVE_DIRECTORY = "//allen/aics/endothelial/morphological_features/IF_integration"
"""Default directory to save IF integration results."""
