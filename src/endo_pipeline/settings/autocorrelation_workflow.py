from enum import StrEnum

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

ACF_CURVE_YLIM = (-0.1, 0.75)

CCF_CURVE_YLIM = (-0.2, 1.0)


class CorrelationDictKeys(StrEnum):
    """Dictionary keys for storing correlation analysis results."""

    TIME_LAGS = "lags"
    """Key for time lags used in correlation calculations."""

    AUTOCORRELATION = "autocorrelation"
    """Key for autocorrelation values."""

    RELAXATION_TIME = "relaxation_timescale"
    """Key for relaxation timescale derived from autocorrelation."""

    CROSS_CORRELATION = "cross_correlation"
    """Key for cross-correlation values."""

    CROSS_CORRELATION_DIFFERENCE = "cross_correlation_difference"
    """Key for difference between forward and backward cross-correlation values."""

    CROSS_CORRELATION_DIFFERENCE_INTEGRAL = "cross_correlation_difference_integral"
    """Key for integral of cross-correlation difference over specified lag range."""

    INTEGRAL_LAG_UPPER_BOUND = "max_lag_integrate"
    """Key for maximum lag used in cross-correlation difference integration."""

    CI_LOWER = "ci_lower"
    """Key modifier for lower bound of confidence interval."""

    CI_UPPER = "ci_upper"
    """Key modifier for upper bound of confidence interval."""
