"""Global settings for image data processing in the endo pipeline."""

DIMENSION_ORDER: str = "TCZYX"
"""Default dimension order of image data."""

ZARR_IMAGE_MANIFEST_NAME: str = "image_zarr"
"""Name of the Zarr image manifest"""

ZARR_BRIGHTFIELD_CHANNEL: int = 1
"""Default channel index for brightfield images in zarr files."""

DIFFAE_ZARR_RESOLUTION_LEVEL: int = 1
"""Default zarr resolution level for loading images for DiffAE model training and inference."""

DIFFAE_DEFAULT_CROP_SIZE: int = 128
"""Default crop size in pixels for DiffAE model training and inference at zarr resolution level 1."""

NATIVE_ZARR_RESOLUTION_CROP_SIZE: int = 256
"""Crop size in pixels at zarr resolution level 0 corresponding to DiffAE_DEFAULT_CROP_SIZE."""

LOWER_Z_SLICE_OFFSET: int = 4
"""How many slices below the "center" Z-plane to include in projections."""

UPPER_Z_SLICE_OFFSET: int = 11
"""How many slices above the "center" Z-plane to include in projections."""

Z_SLICE_OFFSETS: tuple[int, int] = (LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET)
"""Tuple containing the lower and upper Z-offsets for projections."""

LOG_EPSILON: float = 1e-12
"""Small constant used to avoid log(0) errors."""

NUM_ZSLICES: int = 25
"""Number of z-slices per timepoint."""

IF_INTEGRATION_SAVE_DIRECTORY: str = (
    "//allen/aics/endothelial/morphological_features/IF_integration"
)
"""Default production directory to save IF integration results."""

PIXEL_SIZE_3i_20x: float = 0.325
"""Pixel size for the 3i 20x objective in micrometers."""

AXIAL_DISTORTION_CORRECTION_FACTOR_3i_20x: float = 1.43
"""Axial distortion factor for 3i 20x objective determined as described in Diel et al. 2020."""

Z_STEP_SIZE_NOMINAL_3i_20x: float = 0.53
"""Nominal Z-step size for the 20x objective in micrometers."""

Z_STEP_SIZE_ACTUAL_3i_20x: float = (
    Z_STEP_SIZE_NOMINAL_3i_20x * AXIAL_DISTORTION_CORRECTION_FACTOR_3i_20x
)
"""Actual Z-step size for the 20x objective in micrometers, corrected for axial distortion."""
