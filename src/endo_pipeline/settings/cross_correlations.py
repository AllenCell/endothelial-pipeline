"""Method settings for computing the cross correlation function of time series data."""

CROSS_CORR_INDEX_COMBINATIONS = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
"""Feature column index combinations to compute cross-correlations for. The
first three columns are the first three PCs, and the next three columns are the
polar coordinates (radius, polar angle, azimuthal angle)."""

NUM_TIMEPOINT_FRAC = 3
"""Fraction of total number of time points to use as the maximum lag for
cross-correlation and auto-correlation function calculation."""

MAX_LAG_INTEGRATE = 5
"""Maximum lag (in number of time points) to integrate over when calculating the
integral of the difference between positive and negative lags of the CCF."""
