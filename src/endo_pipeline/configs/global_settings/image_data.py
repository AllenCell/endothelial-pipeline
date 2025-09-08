"""Global settings for image data processing in the endo_pipeline."""

LOWER_Z_STACK_OFFSET = 4
"""How many slices below the "center" Z-plane to include in projections."""

UPPER_Z_STACK_OFFSET = 11
"""How many slices above the "center" Z-plane to include in projections."""

Z_STACK_OFFSETS = (LOWER_Z_STACK_OFFSET, UPPER_Z_STACK_OFFSET)
"""Tuple containing the lower and upper Z-offsets for projections."""
