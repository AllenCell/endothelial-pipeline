"""Methods for quantifying the noise signature of drift-removed residuals.

The residuals produced by
:func:`endo_pipeline.library.analyze.numerics.correlations.compute_centered_residuals`
are the part of each observed forward difference that the deterministic drift
does not explain. If the underlying dynamics are an Euler-Maruyama discretized
SDE driven by white noise, those residuals are uncorrelated in time. This module
measures how far they depart from that assumption, and attributes any departure
to either memory in the dynamics or error in the observation of the coordinates.

## Statistics

``lag_one_correlation``
    Normalized residual autocorrelation at a lag of one frame. Zero for white
    noise, positive when the driving noise carries memory, negative when the
    coordinates are observed with independent error.
``integrated_correlation_time``
    Decorrelation time in frames, summed over the initial positive sequence of
    the autocorrelation function. Equal to 0.5 for white noise and larger when
    the noise is persistent.
``spectral_exponent``
    Exponent of a power law fit to the residual power spectrum, where
    ``S(f) ~ f ** -beta``. Zero for white noise, approaching two for noise with
    an Ornstein-Uhlenbeck style correlation, and negative when differencing
    independent observation error dominates.
``measurement_noise_variance``
    Variance of the independent error on each observed coordinate, inferred
    from the lag-one correlation.
``measurement_noise_fraction``
    Share of the residual variance that this error accounts for.
``reliability_ratio``
    Factor by which the same error attenuates a drift estimated by regressing
    displacement on the observed coordinate.

## Two ways to fail whiteness

Whiteness can break in opposite directions, and the two have different causes.
Persistent noise gives ``lag_one_correlation > 0`` and ``spectral_exponent > 0``,
and is what is normally meant by colored noise. Independent error on the
observed coordinates enters the forward difference as
``eta(t) = s(t) + e(t + 1) - e(t)``, where the shared term ``e(t + 1)`` appears
with opposite signs in consecutive residuals. That produces a negative
correlation at exactly lag one, nothing beyond it, and a spectrum that rises
with frequency. Reporting the direction of the departure therefore matters as
much as reporting its size.

## Scope of the permutation null

Every statistic is recalculated on time-permuted copies of the residuals.
Because the permutation preserves the number of patches, the track lengths, and
the positions of missing timepoints, the resulting null accounts for finite
sample effects that asymptotic null distributions ignore. Shuffling within a
patch leaves that patch's mean residual unchanged, so correlations arising from
patch-to-patch heterogeneity rather than from temporal ordering are present in
the surrogates too. The null also holds the empirical residual distribution
fixed, so it does not calibrate away bias introduced by error in the estimated
drift itself.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.noise_signature import (
    NOISE_SIGNATURE_MAX_LAG,
    NOISE_SIGNATURE_MIN_SEGMENT_LENGTH,
    NOISE_SIGNATURE_NUM_SURROGATES,
    NOISE_SIGNATURE_PERCENTILES,
    NOISE_SIGNATURE_RANDOM_SEED,
    NOISE_SIGNATURE_SEGMENT_LENGTH,
)

logger = logging.getLogger(__name__)


class NoiseStatistic(StrEnum):
    """Names of the statistics that make up the noise signature panel."""

    LAG_ONE_CORRELATION = "lag_one_correlation"
    """Normalized residual autocorrelation at a lag of one frame."""

    INTEGRATED_CORRELATION_TIME = "integrated_correlation_time"
    """Decorrelation time of the residuals, in frames."""

    SPECTRAL_EXPONENT = "spectral_exponent"
    """Exponent beta of a power law fit to the residual power spectrum."""

    SPECTRAL_EXPONENT_R_SQUARED = "spectral_exponent_r_squared"
    """Coefficient of determination of the power law fit, in log-log space."""

    RESIDUAL_VARIANCE = "residual_variance"
    """Total variance of the residuals."""

    MEASUREMENT_NOISE_VARIANCE = "measurement_noise_variance"
    """Variance of the independent error on each observed coordinate."""

    MEASUREMENT_NOISE_FRACTION = "measurement_noise_fraction"
    """Share of the residual variance explained by independent measurement error."""

    FEATURE_VARIANCE = "feature_variance"
    """Variance of the observed feature across the population."""

    RELIABILITY_RATIO = "reliability_ratio"
    """Factor by which measurement error attenuates a drift estimated against the feature."""


PERMUTATION_INVARIANT_STATISTICS = frozenset(
    {NoiseStatistic.RESIDUAL_VARIANCE, NoiseStatistic.FEATURE_VARIANCE}
)
"""Statistics that time permutation leaves unchanged and so have no surrogate null."""

SUMMARY_STATISTICS: tuple[NoiseStatistic, ...] = (
    NoiseStatistic.INTEGRATED_CORRELATION_TIME,
    NoiseStatistic.SPECTRAL_EXPONENT,
    NoiseStatistic.SPECTRAL_EXPONENT_R_SQUARED,
    NoiseStatistic.MEASUREMENT_NOISE_VARIANCE,
    NoiseStatistic.RELIABILITY_RATIO,
)
"""Statistics compared across datasets in the summary figure."""

SUMMARY_REFERENCE_VALUES: dict[str, float] = {
    NoiseStatistic.INTEGRATED_CORRELATION_TIME: 0.5,
    NoiseStatistic.SPECTRAL_EXPONENT: 0.0,
    NoiseStatistic.MEASUREMENT_NOISE_VARIANCE: 0.0,
    NoiseStatistic.RELIABILITY_RATIO: 1.0,
}
"""Value each summary statistic takes for white noise observed without error."""


@dataclass
class NoiseSignatureResult:
    """Noise signature statistics together with the curves used to compute them."""

    statistics: pd.DataFrame
    """Tidy dataframe with one row per feature and statistic."""

    lags: np.ndarray
    """Lags, in frames, at which the residual correlations are evaluated."""

    correlations: np.ndarray
    """Normalized residual correlations, of shape (num_lags, num_features, num_features)."""

    correlation_bounds: np.ndarray
    """Surrogate percentile bounds on the diagonal correlations, of shape (2, num_lags, num_features)."""

    frequencies: np.ndarray
    """Frequencies, in inverse frames, at which the power spectrum is evaluated."""

    power_spectrum: np.ndarray
    """Mean residual power spectrum, of shape (num_frequencies, num_features)."""


def compute_lagged_covariances(
    residuals: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pooled lagged covariance matrices of the residuals.

    The covariance at lag ``tau`` is ``R_ij(tau) = < eta_i(t) eta_j(t + tau) >``,
    pooled over every patch and every timepoint at which both members of the
    pair are observed. Pairs that straddle a missing timepoint are excluded
    rather than zero filled, so that gaps in a track do not bias the estimate
    toward zero.

    Parameters
    ----------
    residuals
        Array of shape (num_patches, num_timepoints, num_features) with NaN at
        unobserved timepoints.
    max_lag
        Largest lag, in frames, to evaluate.

    Returns
    -------
    :
        Covariance matrices of shape (max_lag + 1, num_features, num_features).
    :
        Number of contributing pairs at each lag, of shape (max_lag + 1,).
    """
    _, n_timepoints, n_dim = residuals.shape
    max_lag = int(min(max_lag, n_timepoints - 1))

    is_valid = ~np.isnan(residuals).any(axis=2)
    centered = residuals - np.nanmean(residuals, axis=(0, 1))
    # zero fill so that excluded entries contribute nothing to the sums below
    centered = np.where(is_valid[..., np.newaxis], centered, 0.0)

    covariances = np.full((max_lag + 1, n_dim, n_dim), np.nan)
    counts = np.zeros(max_lag + 1, dtype=int)
    for lag in range(max_lag + 1):
        leading = centered[:, : n_timepoints - lag, :]
        trailing = centered[:, lag:, :]
        n_pairs = int((is_valid[:, : n_timepoints - lag] & is_valid[:, lag:]).sum())
        counts[lag] = n_pairs
        if n_pairs == 0:
            continue
        covariances[lag] = np.einsum("pti,ptj->ij", leading, trailing) / n_pairs

    return covariances, counts


def normalize_lagged_covariances(covariances: np.ndarray) -> np.ndarray:
    """Rescale lagged covariances to correlations in [-1, 1]."""
    sigma = np.sqrt(np.diag(covariances[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariances / np.outer(sigma, sigma)


def compute_integrated_correlation_time(correlations: np.ndarray) -> np.ndarray:
    """Compute the decorrelation time of each feature, in frames.

    The sum runs over the initial positive sequence of the autocorrelation
    function, which truncates it before accumulated noise at long lags
    dominates. White noise gives a value of 0.5, and a negative correlation at
    lag one truncates the sum immediately and also gives 0.5.
    """
    n_dim = correlations.shape[1]
    correlation_times = np.full(n_dim, np.nan)
    for index in range(n_dim):
        diagonal = correlations[1:, index, index]
        if diagonal.size == 0 or not np.isfinite(diagonal).any():
            continue
        nonpositive = np.flatnonzero(~(diagonal > 0))
        window = int(nonpositive[0]) if nonpositive.size else diagonal.size
        correlation_times[index] = 0.5 + np.nansum(diagonal[:window])
    return correlation_times


def compute_measurement_noise_variance(
    correlations: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """Estimate the variance of the independent error on each observed coordinate.

    Independent error enters the forward difference as
    ``eta(t) = s(t) + e(t + 1) - e(t)``, which is an MA(1) process with
    ``cov(eta(t), eta(t + 1)) = -var(e)`` and no covariance beyond lag one.
    The error variance is therefore read directly off the lag-one covariance.
    The estimate only has meaning when that covariance is negative, and is
    reported as NaN otherwise.
    """
    if correlations.shape[0] < 2:
        return np.full(correlations.shape[1], np.nan)
    lag_one_correlation = np.einsum("ii->i", correlations[1])
    residual_variance = np.diag(covariances[0])
    return np.where(lag_one_correlation < 0, -lag_one_correlation * residual_variance, np.nan)


def compute_measurement_noise_fraction(correlations: np.ndarray) -> np.ndarray:
    """Estimate the share of residual variance due to independent measurement error.

    The error contributes ``2 * var(e)`` to the residual variance, so the
    fraction is ``-2 * rho(1)``.
    """
    if correlations.shape[0] < 2:
        return np.full(correlations.shape[1], np.nan)
    lag_one_correlation = np.einsum("ii->i", correlations[1])
    return np.where(lag_one_correlation < 0, -2.0 * lag_one_correlation, np.nan)


def compute_reliability_ratio(
    measurement_noise_variance: np.ndarray, feature_variance: np.ndarray
) -> np.ndarray:
    """Compute the factor by which measurement error attenuates an estimated drift.

    Regressing a displacement on a coordinate that is observed with error
    shrinks the fitted slope toward zero by the reliability ratio
    ``var(x_true) / var(x_observed) = 1 - var(e) / var(x_observed)``. A value
    near one means the drift estimate is essentially unbiased; a value well
    below one means the vector field is systematically too flat.

    Parameters
    ----------
    measurement_noise_variance
        Error variance for each feature, of shape (num_features,).
    feature_variance
        Variance of each observed feature across the population, of shape
        (num_features,).

    Returns
    -------
    :
        Reliability ratio for each feature, of shape (num_features,).
    """
    observed_variance = np.asarray(feature_variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.isfinite(measurement_noise_variance) & (observed_variance > 0),
            1.0 - measurement_noise_variance / observed_variance,
            np.nan,
        )


def _extract_fixed_length_segments(residuals: np.ndarray, segment_length: int) -> np.ndarray:
    """Collect non-overlapping gap-free windows of a fixed length from every patch."""
    segments = []
    is_valid = ~np.isnan(residuals).any(axis=2)
    for patch_index in range(residuals.shape[0]):
        valid_flags = is_valid[patch_index]
        # split the patch into maximal runs of consecutive observed timepoints
        boundaries = np.flatnonzero(np.diff(valid_flags.astype(int)) != 0) + 1
        for run in np.split(np.arange(valid_flags.size), boundaries):
            if run.size < segment_length or not valid_flags[run[0]]:
                continue
            for window_index in range(run.size // segment_length):
                start = run[window_index * segment_length]
                segments.append(residuals[patch_index, start : start + segment_length, :])

    if not segments:
        return np.empty((0, segment_length, residuals.shape[2]))
    return np.stack(segments)


def _select_segment_length(residuals: np.ndarray, segment_length: int, min_length: int) -> int:
    """Shrink the requested segment length to a power of two the data can support."""
    is_valid = ~np.isnan(residuals).any(axis=2)
    longest_run = 0
    for patch_index in range(residuals.shape[0]):
        valid_flags = is_valid[patch_index]
        boundaries = np.flatnonzero(np.diff(valid_flags.astype(int)) != 0) + 1
        for run in np.split(np.arange(valid_flags.size), boundaries):
            if run.size and valid_flags[run[0]]:
                longest_run = max(longest_run, int(run.size))

    usable = min(segment_length, longest_run)
    if usable < min_length:
        return 0
    return int(2 ** np.floor(np.log2(usable)))


def compute_power_spectrum(
    residuals: np.ndarray, segment_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the residual power spectrum by averaging periodograms over gap-free segments.

    Parameters
    ----------
    residuals
        Array of shape (num_patches, num_timepoints, num_features).
    segment_length
        Length, in frames, of the segments to transform.

    Returns
    -------
    :
        Positive frequencies in inverse frames, excluding zero and Nyquist.
    :
        Mean power spectrum of shape (num_frequencies, num_features).
    """
    n_dim = residuals.shape[2]
    segments = _extract_fixed_length_segments(residuals, segment_length)
    if segments.shape[0] == 0:
        return np.empty(0), np.empty((0, n_dim))

    segments = segments - segments.mean(axis=1, keepdims=True)
    coefficients = np.fft.rfft(segments, axis=1)
    power = np.abs(coefficients) ** 2 / segment_length
    frequencies = np.fft.rfftfreq(segment_length)

    # the zero frequency is removed by centering and Nyquist has a different
    # sampling distribution, so neither is informative about spectral shape
    keep = slice(1, -1)
    return frequencies[keep], power.mean(axis=0)[keep]


def compute_spectral_exponent(
    frequencies: np.ndarray, power_spectrum: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``S(f) ~ f ** -beta`` to each feature's power spectrum.

    Returns
    -------
    :
        Exponent beta for each feature, of shape (num_features,).
    :
        Coefficient of determination of the fit in log-log space, of shape
        (num_features,). A low value means the spectrum is not well described
        by a single power law, so the exponent should not be read literally.
    """
    n_dim = power_spectrum.shape[1] if power_spectrum.ndim == 2 else 0
    exponents = np.full(n_dim, np.nan)
    r_squared = np.full(n_dim, np.nan)
    if frequencies.size < 3:
        return exponents, r_squared

    log_frequencies = np.log10(frequencies)
    for index in range(n_dim):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_power = np.log10(power_spectrum[:, index])
        finite = np.isfinite(log_power) & np.isfinite(log_frequencies)
        if finite.sum() < 3:
            continue

        slope, intercept = np.polyfit(log_frequencies[finite], log_power[finite], 1)
        exponents[index] = -slope

        predicted = slope * log_frequencies[finite] + intercept
        residual_sum = float(((log_power[finite] - predicted) ** 2).sum())
        total_sum = float(((log_power[finite] - log_power[finite].mean()) ** 2).sum())
        if total_sum > 0:
            r_squared[index] = 1.0 - residual_sum / total_sum

    return exponents, r_squared


def permute_residuals_in_time(residuals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle residual vectors along time within each patch, preserving the missing data mask.

    Permuting whole vectors keeps the instantaneous cross-feature covariance and
    the per-patch amplitude intact while destroying any temporal ordering, which
    makes the result a null sample for every whiteness statistic in this module.
    """
    is_valid = ~np.isnan(residuals).any(axis=2)

    # sorting random keys puts a random permutation of the observed indices
    # first, while sorting the mask puts the observed indices themselves first
    keys = rng.random(is_valid.shape)
    keys[~is_valid] = np.inf
    source = np.argsort(keys, axis=1)
    target = np.argsort((~is_valid).astype(int), kind="stable", axis=1)

    rows = np.arange(residuals.shape[0])[:, np.newaxis]
    permuted = np.empty_like(residuals)
    permuted[rows, target] = residuals[rows, source]
    return permuted


def _compute_statistic_values(
    residuals: np.ndarray,
    column_names: list[str],
    max_lag: int,
    segment_length: int,
    feature_variance: np.ndarray | None,
) -> tuple[dict[tuple[str, str], float], np.ndarray, np.ndarray]:
    """Compute every noise signature statistic for one residual array."""
    covariances, _ = compute_lagged_covariances(residuals, max_lag)
    correlations = normalize_lagged_covariances(covariances)

    correlation_times = compute_integrated_correlation_time(correlations)
    noise_variance = compute_measurement_noise_variance(correlations, covariances)
    noise_fraction = compute_measurement_noise_fraction(correlations)
    residual_variance = np.diag(covariances[0])
    lag_one = (
        np.einsum("ii->i", correlations[1])
        if correlations.shape[0] > 1
        else np.full(len(column_names), np.nan)
    )

    if segment_length > 0:
        frequencies, power_spectrum = compute_power_spectrum(residuals, segment_length)
    else:
        frequencies, power_spectrum = np.empty(0), np.empty((0, len(column_names)))
    spectral_exponent, spectral_r_squared = compute_spectral_exponent(frequencies, power_spectrum)

    observed_variance = np.full(len(column_names), np.nan)
    reliability_ratio = np.full(len(column_names), np.nan)
    if feature_variance is not None:
        observed_variance = np.asarray(feature_variance, dtype=float)
        reliability_ratio = compute_reliability_ratio(noise_variance, observed_variance)

    statistic_values = {
        NoiseStatistic.LAG_ONE_CORRELATION: lag_one,
        NoiseStatistic.INTEGRATED_CORRELATION_TIME: correlation_times,
        NoiseStatistic.SPECTRAL_EXPONENT: spectral_exponent,
        NoiseStatistic.SPECTRAL_EXPONENT_R_SQUARED: spectral_r_squared,
        NoiseStatistic.RESIDUAL_VARIANCE: residual_variance,
        NoiseStatistic.MEASUREMENT_NOISE_VARIANCE: noise_variance,
        NoiseStatistic.MEASUREMENT_NOISE_FRACTION: noise_fraction,
        NoiseStatistic.FEATURE_VARIANCE: observed_variance,
        NoiseStatistic.RELIABILITY_RATIO: reliability_ratio,
    }

    values: dict[tuple[str, str], float] = {}
    for statistic, per_feature in statistic_values.items():
        for index, column_name in enumerate(column_names):
            values[(column_name, statistic)] = float(per_feature[index])

    return values, correlations, power_spectrum


def compute_noise_signature(
    residuals: np.ndarray,
    column_names: list[str],
    max_lag: int = NOISE_SIGNATURE_MAX_LAG,
    num_surrogates: int = NOISE_SIGNATURE_NUM_SURROGATES,
    segment_length: int = NOISE_SIGNATURE_SEGMENT_LENGTH,
    random_seed: int = NOISE_SIGNATURE_RANDOM_SEED,
    percentiles: tuple[float, float] = NOISE_SIGNATURE_PERCENTILES,
    feature_variance: np.ndarray | None = None,
    metadata_dict: dict[str, str | float] | None = None,
) -> NoiseSignatureResult | None:
    """Quantify how far drift-removed residuals depart from white noise.

    Parameters
    ----------
    residuals
        Array of shape (num_patches, num_timepoints, num_features) with NaN at
        unobserved timepoints.
    column_names
        List of feature column names, of length num_features.
    max_lag
        Largest lag, in frames, used by the correlation statistics.
    num_surrogates
        Number of time-permuted surrogate datasets used to calibrate the statistics.
    segment_length
        Target length, in frames, of the gap-free segments used for spectral estimation.
    random_seed
        Seed for the surrogate random number generator.
    percentiles
        Lower and upper percentiles of the surrogate distribution used for the
        correlation confidence band.
    feature_variance
        Optional variance of each observed feature across the population, of
        shape (num_features,), required for the reliability ratio.
    metadata_dict
        Optional dictionary of additional metadata to add as columns to the
        output dataframe.

    Returns
    -------
    :
        Noise signature statistics and the curves used to compute them, or None
        if the residuals contain too few observations to analyze.
    """
    if residuals.ndim != 3 or residuals.shape[2] != len(column_names):
        logger.error(
            "Residual array of shape %s is incompatible with %d feature names.",
            residuals.shape,
            len(column_names),
        )
        return None

    n_dim = len(column_names)
    max_lag = int(min(max_lag, residuals.shape[1] - 1))
    if max_lag < 1:
        logger.warning("Too few timepoints to compute noise signature statistics. Skipping.")
        return None

    usable_segment_length = _select_segment_length(
        residuals, segment_length, NOISE_SIGNATURE_MIN_SEGMENT_LENGTH
    )
    if usable_segment_length == 0:
        logger.warning(
            "No gap-free segment of at least %d frames found. "
            "The spectral exponent will not be computed.",
            NOISE_SIGNATURE_MIN_SEGMENT_LENGTH,
        )
    elif usable_segment_length < segment_length:
        logger.debug(
            "Reducing spectral segment length from %d to %d frames.",
            segment_length,
            usable_segment_length,
        )

    values, correlations, power_spectrum = _compute_statistic_values(
        residuals, column_names, max_lag, usable_segment_length, feature_variance
    )

    rng = np.random.default_rng(random_seed)
    surrogate_values: dict[tuple[str, str], list[float]] = {key: [] for key in values}
    surrogate_correlations = np.full((num_surrogates, max_lag + 1, n_dim), np.nan)
    for surrogate_index in range(num_surrogates):
        permuted = permute_residuals_in_time(residuals, rng)
        permuted_values, permuted_correlations, _ = _compute_statistic_values(
            permuted, column_names, max_lag, usable_segment_length, feature_variance
        )
        for key, value in permuted_values.items():
            surrogate_values[key].append(value)
        surrogate_correlations[surrogate_index] = np.einsum("lii->li", permuted_correlations)

    records = []
    for (feature, statistic), value in values.items():
        samples = np.asarray(surrogate_values[(feature, statistic)], dtype=float)
        finite_samples = samples[np.isfinite(samples)]
        if statistic in PERMUTATION_INVARIANT_STATISTICS:
            finite_samples = np.empty(0)

        surrogate_mean = float("nan")
        surrogate_std = float("nan")
        if finite_samples.size:
            surrogate_mean = float(finite_samples.mean())
        if finite_samples.size > 1:
            surrogate_std = float(finite_samples.std(ddof=1))

        z_score = float("nan")
        if np.isfinite(surrogate_std) and surrogate_std > 0:
            z_score = (value - surrogate_mean) / surrogate_std

        # two-sided empirical p-value, offset by one to keep it strictly positive
        p_value_permutation = float("nan")
        if finite_samples.size and np.isfinite(value):
            n_extreme = int(
                (np.abs(finite_samples - surrogate_mean) >= abs(value - surrogate_mean)).sum()
            )
            p_value_permutation = (1 + n_extreme) / (1 + finite_samples.size)

        records.append(
            {
                Column.NoiseSignature.FEATURE: str(feature),
                Column.NoiseSignature.STATISTIC: str(statistic),
                Column.NoiseSignature.VALUE: value,
                Column.NoiseSignature.SURROGATE_MEAN: surrogate_mean,
                Column.NoiseSignature.SURROGATE_STD: surrogate_std,
                Column.NoiseSignature.Z_SCORE: z_score,
                Column.NoiseSignature.P_VALUE_PERMUTATION: p_value_permutation,
            }
        )

    statistics = pd.DataFrame.from_records(records)
    if metadata_dict is not None:
        for key in metadata_dict:
            statistics[key] = metadata_dict[key]

    correlation_bounds = np.full((2, max_lag + 1, n_dim), np.nan)
    if num_surrogates > 1:
        correlation_bounds = np.nanpercentile(surrogate_correlations, percentiles, axis=0)

    frequencies = (
        np.fft.rfftfreq(usable_segment_length)[1:-1] if usable_segment_length > 0 else np.empty(0)
    )

    return NoiseSignatureResult(
        statistics=statistics,
        lags=np.arange(max_lag + 1),
        correlations=correlations,
        correlation_bounds=correlation_bounds,
        frequencies=frequencies,
        power_spectrum=power_spectrum,
    )
