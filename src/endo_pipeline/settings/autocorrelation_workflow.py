"""Workflow settings for autocorrelation analysis in endo_pipeline."""

CROSS_CORR_INDEX_COMBINATIONS = [(0, 1), (0, 2), (1, 2)]
"""Index combinations for cross-correlating feature pairs."""

NUM_TIMEPOINT_FRAC = 3
"""Fraction of total timepoints to define lag range for auto- and cross-correlation calculations."""

MAX_LAG_INTEGRATE = 5
"""Maximum lag (in timepoints) for integrating forward and backward cross-correlation differences."""

NUM_BOOTSTRAP_SAMPLES = 1000
"""Number of bootstrap samples for estimating confidence intervals in correlation analyses."""

CONFIDENCE_LEVEL = 0.95
"""Confidence level for bootstrap confidence intervals."""
