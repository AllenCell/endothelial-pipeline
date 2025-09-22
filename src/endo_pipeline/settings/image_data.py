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

AXIAL_DISTORTION_CORRECTION_FACTOR_3i_20x = 1.43
"""Axial distortion factor for 3i 20x objective determined as described in Diel et al. 2020."""

AXIAL_DISTORTION_CORRECTION_FACTOR_3i_40x = 1.00
"""Axial distortion factor for 3i 40x objective determined as described in Diel et al. 2020."""

Z_STEP_SIZE_NOMINAL_3i_20x = 0.53
"""Nominal Z-step size for the 20x objective in micrometers."""

Z_STEP_SIZE_NOMINAL_3i_40x = 0.26
"""Nominal Z-step size for the 40x objective in micrometers."""

Z_STEP_SIZE_ACTUAL_3i_20x = Z_STEP_SIZE_NOMINAL_3i_20x * AXIAL_DISTORTION_CORRECTION_FACTOR_3i_20x
"""Actual Z-step size for the 20x objective in micrometers, corrected for axial distortion."""

Z_STEP_SIZE_ACTUAL_3i_40x = Z_STEP_SIZE_NOMINAL_3i_40x * AXIAL_DISTORTION_CORRECTION_FACTOR_3i_40x
"""Actual Z-step size for the 40x objective in micrometers, corrected for axial distortion."""
