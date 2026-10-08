"""Method settings for quantifying the noise signature of drift-removed residuals."""

NOISE_SIGNATURE_MAX_LAG: int = 10
"""Maximum lag (in frames) used for the residual autocorrelation and portmanteau statistics."""

NOISE_SIGNATURE_NUM_SURROGATES: int = 200
"""Number of time-permuted surrogate datasets used to calibrate the whiteness statistics."""

NOISE_SIGNATURE_SEGMENT_LENGTH: int = 32
"""Target length (in frames) of the contiguous segments used to estimate the power spectrum."""

NOISE_SIGNATURE_MIN_SEGMENT_LENGTH: int = 8
"""Shortest segment length accepted for spectral estimation before it is skipped."""

NOISE_SIGNATURE_RANDOM_SEED: int = 0
"""Random seed used to generate surrogate datasets."""

NOISE_SIGNATURE_PERCENTILES: tuple[float, float] = (2.5, 97.5)
"""Percentiles of the surrogate distribution used to draw whiteness confidence bands."""

NOISE_SIGNATURE_FMS_ANNOTATION_NOTES: str = (
    "Dataframe containing noise signature statistics for drift-removed residuals."
)
"""Annotation notes for noise signature dataframes uploaded to FMS."""
