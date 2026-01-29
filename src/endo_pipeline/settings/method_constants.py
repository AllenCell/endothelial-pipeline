OUTLIER_THRESHOLD = 0.01
"""Percentage to use for thresholding dark and bright gfp and bf outliers."""

PARTIAL_DARK_THRESHOLD = 0.004
"""Percentage to use for thresholding partial dark bf outliers."""

GFP_ROLLING_WINDOW = 12
"""Number of timepoints to use for rolling window calculation of gfp outliers (1 hour)."""

BF_ROLLING_WINDOW = 100
"""Number of z-slices per to use for rolling window calculation of bf outliers (4 timepoints)."""
