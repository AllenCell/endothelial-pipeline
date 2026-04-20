"""Method settings for computing the cross correlation function of time series data."""

NUM_TIMEPOINT_FRAC = 3
"""Fraction of total number of time points to use as the maximum lag for
cross-correlation and auto-correlation function calculation."""

MAX_LAG_INTEGRATE = 5
"""Maximum lag (in number of time points) to integrate over when calculating the
integral of the difference between positive and negative lags of the CCF."""
