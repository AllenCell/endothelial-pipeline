"""Metrics and tests that quantify whether a noise residual series is white."""

import logging
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstwobign

from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.noise_whiteness import (
    MAX_LAG_FRACTION,
    MAX_LAG_INTEGRATED_TIME,
    MAX_LAG_WHITENESS,
    NUM_LAGS_PORTMANTEAU,
    NUM_SURROGATES,
    SPECTRAL_FIT_FREQUENCY_RANGE,
    SURROGATE_PERCENTILES,
    SURROGATE_RANDOM_SEED,
)

logger = logging.getLogger(__name__)

ALL_FEATURES_LABEL = "all"
"""Feature label used for statistics computed jointly across all features."""

MIN_POINTS_FOR_SPECTRAL_FIT = 4
"""Minimum number of frequency bins required to fit a spectral exponent."""


class NoiseWhitenessMetric(StrEnum):
    """Scalar statistics summarizing the noise signature of a residual series."""

    LAG_ONE_CORRELATION = "lag_one_correlation"
    """Normalized residual autocorrelation at lag 1; zero for white noise."""

    INTEGRATED_CORRELATION_TIME = "integrated_correlation_time"
    """Integrated correlation time in frames; one for white noise, larger for
    positively correlated (red) noise and smaller for anticorrelated noise."""

    BOX_PIERCE_Q = "box_pierce_q"
    """Per-feature portmanteau statistic testing that all lags up to the cutoff
    have zero autocorrelation."""

    HOSKING_Q = "hosking_q"
    """Multivariate portmanteau statistic jointly testing all features and lags."""

    SPECTRAL_EXPONENT = "spectral_exponent"
    """Exponent of a power-law fit to the residual power spectrum, where
    `S(f) ~ f^-beta`; zero for white noise and approaching two for
    Ornstein-Uhlenbeck-like colored noise."""

    BARTLETT_KS = "bartlett_ks"
    """Kolmogorov-Smirnov distance between the normalized cumulative periodogram
    and the straight line expected for a flat (white) spectrum."""

    EXCESS_KURTOSIS = "excess_kurtosis"
    """Excess kurtosis of the standardized residuals; non-zero values indicate
    non-Gaussian noise such as jumps."""

    SQUARED_RESIDUAL_LAG_ONE = "squared_residual_lag_one"
    """Lag-1 autocorrelation of the squared residuals. Non-zero values indicate
    structured volatility (multiplicative noise) even when the residuals
    themselves look white."""


UPPER_TAIL_METRICS = frozenset(
    {
        NoiseWhitenessMetric.BOX_PIERCE_Q,
        NoiseWhitenessMetric.HOSKING_Q,
        NoiseWhitenessMetric.BARTLETT_KS,
    }
)
"""Metrics for which only large values are evidence against white noise."""


def get_max_lag(num_timepoints: int, max_lag: int = MAX_LAG_WHITENESS) -> int:
    """Cap the requested maximum lag so that each lag retains enough sample pairs."""
    return max(1, min(max_lag, num_timepoints // MAX_LAG_FRACTION))


def compute_lagged_noise_covariances(
    residuals: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pooled lagged covariance matrices of the noise residuals.

    The lagged covariance is

        Gamma_ij(tau) = < eta_i(t) eta_j(t + tau) >,

    pooled over all patches and all timepoints at which both members of the pair
    are observed. Pairs involving an unobserved timepoint are excluded rather
    than zero-filled, so short and gappy patches do not bias the estimate toward
    zero.

    Parameters
    ----------
    residuals
        Array of shape (num_patches, num_timepoints, num_features) containing
        the noise residuals, with NaN at unobserved timepoints.
    max_lag
        Largest lag, in frames, at which to evaluate the covariance.

    Returns
    -------
    :
        Tuple of the lagged covariances, of shape (max_lag + 1, num_features,
        num_features), and the number of sample pairs contributing to each lag,
        of shape (max_lag + 1,).
    """
    num_timepoints = residuals.shape[1]
    num_features = residuals.shape[2]

    is_valid = ~np.isnan(residuals).any(axis=2)
    num_valid = int(is_valid.sum())
    if num_valid == 0:
        raise ValueError("Residual array contains no valid timepoints.")

    # center on the pooled mean so that any residual bias in the drift estimate
    # does not masquerade as correlation
    filled = np.where(is_valid[:, :, np.newaxis], residuals, 0.0)
    pooled_mean = filled.sum(axis=(0, 1)) / num_valid
    centered = np.where(is_valid[:, :, np.newaxis], residuals - pooled_mean, 0.0)

    covariances = np.full((max_lag + 1, num_features, num_features), np.nan)
    counts = np.zeros(max_lag + 1, dtype=int)

    for lag in range(max_lag + 1):
        leading = centered[:, : num_timepoints - lag, :]
        trailing = centered[:, lag:, :]
        pair_is_valid = is_valid[:, : num_timepoints - lag] & is_valid[:, lag:]
        num_pairs = int(pair_is_valid.sum())
        counts[lag] = num_pairs
        if num_pairs > 0:
            covariances[lag] = np.einsum("pti,ptj->ij", leading, trailing) / num_pairs

    return covariances, counts


def normalize_lagged_covariances(covariances: np.ndarray) -> np.ndarray:
    """Rescale lagged covariances to correlations using the lag-zero variances."""
    variances = np.diag(covariances[0])
    scale = np.sqrt(np.outer(variances, variances))
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariances / scale[np.newaxis, :, :]


def compute_integrated_correlation_time(
    correlations: np.ndarray, max_lag_integrate: int = MAX_LAG_INTEGRATED_TIME
) -> np.ndarray:
    """
    Compute the integrated correlation time of each feature, in frames.

    Uses the convention `tau_int = 1 + 2 * sum_{tau >= 1} rho(tau)`, so that
    white noise gives `tau_int = 1`.

    Parameters
    ----------
    correlations
        Normalized lagged correlations of shape (num_lags, num_features,
        num_features).
    max_lag_integrate
        Number of positive lags included in the sum.

    Returns
    -------
    :
        Array of shape (num_features,) containing the integrated correlation time.
    """
    num_lags = min(max_lag_integrate, correlations.shape[0] - 1)
    diagonal = np.einsum("lii->li", correlations[: num_lags + 1])
    return 1.0 + 2.0 * np.nansum(diagonal[1:], axis=0)


def compute_box_pierce_statistic(
    correlations: np.ndarray, counts: np.ndarray, num_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a per-feature portmanteau statistic and its asymptotic p-value.

    Uses the Box-Pierce form weighted by the number of sample pairs available at
    each lag, `Q = sum_tau n(tau) * rho(tau)^2`, which is asymptotically
    chi-squared with `num_lags` degrees of freedom under white noise. The
    per-lag weighting is used in place of the usual Ljung-Box correction because
    the residuals are pooled across patches of unequal and gappy length.

    Parameters
    ----------
    correlations
        Normalized lagged correlations of shape (num_lags, num_features,
        num_features).
    counts
        Number of sample pairs at each lag.
    num_lags
        Number of positive lags included in the statistic.

    Returns
    -------
    :
        Tuple of the statistic and the asymptotic p-value, each of shape
        (num_features,).
    """
    num_lags = min(num_lags, correlations.shape[0] - 1)
    diagonal = np.einsum("lii->li", correlations[: num_lags + 1])
    statistic = np.nansum(counts[1 : num_lags + 1, np.newaxis] * diagonal[1:] ** 2, axis=0)
    return statistic, chi2.sf(statistic, df=num_lags)


def compute_hosking_statistic(
    covariances: np.ndarray, counts: np.ndarray, num_lags: int
) -> tuple[float, float]:
    """
    Compute the multivariate portmanteau statistic and its asymptotic p-value.

    The statistic is

        Q = sum_tau n(tau) * tr(Gamma_tau^T Gamma_0^-1 Gamma_tau Gamma_0^-1),

    which is asymptotically chi-squared with `num_features^2 * num_lags` degrees
    of freedom under multivariate white noise. Unlike the per-feature statistic
    it is also sensitive to cross-feature lagged structure, where feature i at
    time t predicts feature j at time t + tau.

    Parameters
    ----------
    covariances
        Lagged covariances of shape (num_lags, num_features, num_features).
    counts
        Number of sample pairs at each lag.
    num_lags
        Number of positive lags included in the statistic.

    Returns
    -------
    :
        Tuple of the statistic and the asymptotic p-value.
    """
    num_features = covariances.shape[1]
    num_lags = min(num_lags, covariances.shape[0] - 1)

    # pseudo-inverse guards against a singular lag-zero covariance, which occurs
    # when two features are degenerate
    inverse_gamma_zero = np.linalg.pinv(covariances[0])

    statistic = 0.0
    for lag in range(1, num_lags + 1):
        gamma = covariances[lag]
        if not np.isfinite(gamma).all():
            continue
        product = gamma.T @ inverse_gamma_zero @ gamma @ inverse_gamma_zero
        statistic += float(counts[lag]) * float(np.trace(product))

    degrees_of_freedom = num_features**2 * num_lags
    return statistic, float(chi2.sf(statistic, df=degrees_of_freedom))


def compute_mean_periodogram(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the patch-averaged power spectral density of the noise residuals.

    Unobserved timepoints are zero-filled after centering, which biases the
    spectrum toward flatness. The surrogate residuals carry exactly the same
    gap structure, so the bias cancels when the spectral statistics are
    calibrated against them.

    Parameters
    ----------
    residuals
        Array of shape (num_patches, num_timepoints, num_features).

    Returns
    -------
    :
        Tuple of the frequencies, in cycles per frame and excluding the zero
        frequency, of shape (num_frequencies,), and the power spectral density
        of shape (num_frequencies, num_features).
    """
    num_timepoints = residuals.shape[1]

    is_valid = ~np.isnan(residuals).any(axis=2)
    num_valid_per_patch = is_valid.sum(axis=1)

    filled = np.where(is_valid[:, :, np.newaxis], residuals, 0.0)
    num_valid = int(is_valid.sum())
    pooled_mean = filled.sum(axis=(0, 1)) / num_valid
    centered = np.where(is_valid[:, :, np.newaxis], residuals - pooled_mean, 0.0)

    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    # normalize per patch by its own number of observations so that long patches
    # do not dominate the average
    has_data = num_valid_per_patch > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        spectrum = spectrum[has_data] / num_valid_per_patch[has_data, np.newaxis, np.newaxis]

    frequencies = np.fft.rfftfreq(num_timepoints, d=1.0)
    return frequencies[1:], spectrum.mean(axis=0)[1:]


def fit_spectral_exponent(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    frequency_range: tuple[float, float] = SPECTRAL_FIT_FREQUENCY_RANGE,
) -> np.ndarray:
    """
    Fit the power-law exponent `beta` of `S(f) ~ f^-beta` for each feature.

    Parameters
    ----------
    frequencies
        Frequencies in cycles per frame, of shape (num_frequencies,).
    power_spectrum
        Power spectral density of shape (num_frequencies, num_features).
    frequency_range
        Lower and upper frequency bounds of the fit.

    Returns
    -------
    :
        Array of shape (num_features,) containing the spectral exponent, or NaN
        where too few frequency bins fall inside the fit range.
    """
    num_features = power_spectrum.shape[1]
    in_range = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])

    if int(in_range.sum()) < MIN_POINTS_FOR_SPECTRAL_FIT:
        logger.warning(
            "Too few frequency bins in range %s to fit a spectral exponent.", frequency_range
        )
        return np.full(num_features, np.nan)

    log_frequencies = np.log(frequencies[in_range])
    exponents = np.full(num_features, np.nan)
    for feature_index in range(num_features):
        power = power_spectrum[in_range, feature_index]
        if not np.all(power > 0):
            continue
        slope, _ = np.polyfit(log_frequencies, np.log(power), deg=1)
        exponents[feature_index] = -slope

    return exponents


def compute_bartlett_test(power_spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Test the flatness of the residual spectrum via the cumulative periodogram.

    For white noise the normalized cumulative periodogram rises linearly from
    zero to one, so the Kolmogorov-Smirnov distance from that straight line
    measures departure from whiteness. This is Bartlett's test, and unlike the
    portmanteau statistics it is sensitive to broadband correlation spread
    thinly across many lags.

    Parameters
    ----------
    power_spectrum
        Power spectral density of shape (num_frequencies, num_features),
        excluding the zero frequency.

    Returns
    -------
    :
        Tuple of the Kolmogorov-Smirnov distance and its asymptotic p-value,
        each of shape (num_features,).
    """
    num_frequencies, num_features = power_spectrum.shape
    totals = power_spectrum.sum(axis=0)

    distances = np.full(num_features, np.nan)
    for feature_index in range(num_features):
        if not np.isfinite(totals[feature_index]) or totals[feature_index] <= 0:
            continue
        cumulative = np.cumsum(power_spectrum[:, feature_index]) / totals[feature_index]
        expected = np.arange(1, num_frequencies + 1) / num_frequencies
        distances[feature_index] = np.max(np.abs(cumulative - expected))

    p_values = kstwobign.sf(np.sqrt(num_frequencies) * distances)
    return distances, p_values


def permute_residuals_in_time(residuals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Build a surrogate residual array by shuffling each patch in time.

    Shuffling destroys temporal correlation while exactly preserving the number
    of patches, the per-patch observation mask, and the marginal distribution of
    residual amplitudes. The resulting null distribution therefore absorbs the
    biases introduced by short tracks, missing timepoints, and zero-filling,
    which the asymptotic reference distributions do not account for.

    Parameters
    ----------
    residuals
        Array of shape (num_patches, num_timepoints, num_features).
    rng
        Random number generator used to draw the permutations.

    Returns
    -------
    :
        Surrogate residual array with the same shape and NaN pattern as the input.
    """
    surrogate = np.full_like(residuals, np.nan)
    is_valid = ~np.isnan(residuals).any(axis=2)

    for patch_index in range(residuals.shape[0]):
        valid_indices = np.flatnonzero(is_valid[patch_index])
        surrogate[patch_index, valid_indices] = residuals[
            patch_index, rng.permutation(valid_indices)
        ]

    return surrogate


def compute_whiteness_statistics(
    residuals: np.ndarray,
    column_names: list[str],
    max_lag: int,
    num_lags_portmanteau: int = NUM_LAGS_PORTMANTEAU,
    max_lag_integrate: int = MAX_LAG_INTEGRATED_TIME,
) -> tuple[pd.Series, np.ndarray, pd.Series]:
    """
    Compute the full panel of scalar whiteness statistics for a residual array.

    Parameters
    ----------
    residuals
        Array of shape (num_patches, num_timepoints, num_features).
    column_names
        Feature names corresponding to the last axis of `residuals`.
    max_lag
        Largest lag at which lagged correlations are evaluated.
    num_lags_portmanteau
        Number of lags included in the portmanteau statistics.
    max_lag_integrate
        Number of lags summed for the integrated correlation time.

    Returns
    -------
    :
        Tuple of the statistics, indexed by (metric, feature); the normalized
        lagged correlations of shape (max_lag + 1, num_features, num_features);
        and the asymptotic p-values, indexed by (metric, feature).
    """
    covariances, counts = compute_lagged_noise_covariances(residuals, max_lag)
    correlations = normalize_lagged_covariances(covariances)

    integrated_time = compute_integrated_correlation_time(correlations, max_lag_integrate)
    box_pierce, box_pierce_p = compute_box_pierce_statistic(
        correlations, counts, num_lags_portmanteau
    )
    hosking, hosking_p = compute_hosking_statistic(covariances, counts, num_lags_portmanteau)

    frequencies, power_spectrum = compute_mean_periodogram(residuals)
    spectral_exponent = fit_spectral_exponent(frequencies, power_spectrum)
    bartlett_ks, bartlett_p = compute_bartlett_test(power_spectrum)

    excess_kurtosis = _compute_excess_kurtosis(residuals)
    squared_covariances, _ = compute_lagged_noise_covariances(_square_residuals(residuals), max_lag)
    squared_lag_one = np.einsum("ii->i", normalize_lagged_covariances(squared_covariances)[1])

    per_feature = {
        NoiseWhitenessMetric.LAG_ONE_CORRELATION: np.einsum("ii->i", correlations[1]),
        NoiseWhitenessMetric.INTEGRATED_CORRELATION_TIME: integrated_time,
        NoiseWhitenessMetric.BOX_PIERCE_Q: box_pierce,
        NoiseWhitenessMetric.SPECTRAL_EXPONENT: spectral_exponent,
        NoiseWhitenessMetric.BARTLETT_KS: bartlett_ks,
        NoiseWhitenessMetric.EXCESS_KURTOSIS: excess_kurtosis,
        NoiseWhitenessMetric.SQUARED_RESIDUAL_LAG_ONE: squared_lag_one,
    }
    per_feature_p_values = {
        NoiseWhitenessMetric.BOX_PIERCE_Q: box_pierce_p,
        NoiseWhitenessMetric.BARTLETT_KS: bartlett_p,
    }

    index = []
    values = []
    p_values = []
    for metric, metric_values in per_feature.items():
        metric_p_values = per_feature_p_values.get(metric)
        for feature_index, column_name in enumerate(column_names):
            index.append((str(metric), str(column_name)))
            values.append(float(metric_values[feature_index]))
            p_values.append(
                np.nan if metric_p_values is None else float(metric_p_values[feature_index])
            )

    index.append((str(NoiseWhitenessMetric.HOSKING_Q), ALL_FEATURES_LABEL))
    values.append(hosking)
    p_values.append(hosking_p)

    multi_index = pd.MultiIndex.from_tuples(
        index, names=[Column.NoiseWhiteness.METRIC, Column.NoiseWhiteness.FEATURE]
    )
    return (
        pd.Series(values, index=multi_index),
        correlations,
        pd.Series(p_values, index=multi_index),
    )


def _square_residuals(residuals: np.ndarray) -> np.ndarray:
    """Square the residuals, preserving the NaN pattern."""
    return residuals**2


def _compute_excess_kurtosis(residuals: np.ndarray) -> np.ndarray:
    """Compute the excess kurtosis of the pooled residuals for each feature."""
    flattened = residuals.reshape(-1, residuals.shape[2])
    mean = np.nanmean(flattened, axis=0)
    standard_deviation = np.nanstd(flattened, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = (flattened - mean) / standard_deviation
    return np.nanmean(standardized**4, axis=0) - 3.0


def compute_noise_whiteness_dataframes(
    residuals: np.ndarray,
    column_names: list[str],
    num_surrogates: int = NUM_SURROGATES,
    random_seed: int = SURROGATE_RANDOM_SEED,
    max_lag: int = MAX_LAG_WHITENESS,
    metadata_dict: dict[str, str | float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quantify the noise signature of a residual series against a surrogate null.

    Every statistic is recomputed on `num_surrogates` time-shuffled copies of the
    residuals. The shuffles share the observation mask, patch count, and
    amplitude distribution of the data but have no temporal structure, so they
    give the distribution each statistic would take under exactly white noise
    observed through the same sampling geometry. Reported z-scores and empirical
    p-values are taken against that distribution rather than against the
    asymptotic reference distributions, which assume regular, gap-free sampling
    and a known drift.

    Note that the surrogates do not absorb the spurious negative lag-one
    correlation induced by Euler-Maruyama discretization error in the drift
    estimate, since shuffling removes that structure along with any real
    correlation. A non-zero lag-one correlation should therefore be interpreted
    alongside the sign expected from discretization, which is negative.

    Parameters
    ----------
    residuals
        Array of shape (num_patches, num_timepoints, num_features) containing
        the noise residuals, with NaN at unobserved timepoints.
    column_names
        Feature names corresponding to the last axis of `residuals`.
    num_surrogates
        Number of time-shuffled surrogates used to build the null distribution.
    random_seed
        Seed for the random number generator used to draw surrogates.
    max_lag
        Largest lag at which lagged correlations are evaluated.
    metadata_dict
        Optional dictionary of additional metadata to add as columns to both
        output dataframes.

    Returns
    -------
    :
        Tuple of the scalar metric dataframe, with one row per metric and
        feature, and the lagged correlation dataframe, with one row per feature
        pair and lag.
    """
    max_lag = get_max_lag(residuals.shape[1], max_lag)

    observed, correlations, analytic_p_values = compute_whiteness_statistics(
        residuals, column_names, max_lag
    )

    rng = np.random.default_rng(random_seed)
    surrogate_statistics = []
    surrogate_correlations = []
    for _ in range(num_surrogates):
        surrogate_residuals = permute_residuals_in_time(residuals, rng)
        statistics, surrogate_rho, _ = compute_whiteness_statistics(
            surrogate_residuals, column_names, max_lag
        )
        surrogate_statistics.append(statistics)
        surrogate_correlations.append(surrogate_rho)

    null_distribution = pd.concat(surrogate_statistics, axis=1).to_numpy()
    null_correlations = np.stack(surrogate_correlations)

    metric_dataframe = _build_metric_dataframe(observed, analytic_p_values, null_distribution)
    metric_dataframe[Column.NoiseWhiteness.NUM_SAMPLES] = int(
        (~np.isnan(residuals).any(axis=2)).sum()
    )
    correlation_dataframe = _build_correlation_dataframe(
        correlations, null_correlations, column_names
    )

    if metadata_dict is not None:
        for key, value in metadata_dict.items():
            metric_dataframe[key] = value
            correlation_dataframe[key] = value

    return metric_dataframe, correlation_dataframe


def _build_metric_dataframe(
    observed: pd.Series, analytic_p_values: pd.Series, null_distribution: np.ndarray
) -> pd.DataFrame:
    """Assemble the scalar metric dataframe with surrogate-calibrated significance."""
    observed_values = observed.to_numpy()
    surrogate_mean = np.nanmean(null_distribution, axis=1)
    surrogate_std = np.nanstd(null_distribution, axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = (observed_values - surrogate_mean) / surrogate_std

    num_surrogates = null_distribution.shape[1]
    is_upper_tail = np.array(
        [metric in UPPER_TAIL_METRICS for metric, _ in observed.index], dtype=bool
    )
    # upper-tail metrics are only extreme when large; the rest are two-sided
    # about the surrogate mean
    exceedances = np.where(
        is_upper_tail[:, np.newaxis],
        null_distribution >= observed_values[:, np.newaxis],
        np.abs(null_distribution - surrogate_mean[:, np.newaxis])
        >= np.abs(observed_values - surrogate_mean)[:, np.newaxis],
    )
    surrogate_p_values = (1.0 + np.nansum(exceedances, axis=1)) / (num_surrogates + 1.0)

    return pd.DataFrame(
        {
            Column.NoiseWhiteness.METRIC: observed.index.get_level_values(0),
            Column.NoiseWhiteness.FEATURE: observed.index.get_level_values(1),
            Column.NoiseWhiteness.VALUE: observed_values,
            Column.NoiseWhiteness.SURROGATE_MEAN: surrogate_mean,
            Column.NoiseWhiteness.SURROGATE_STD: surrogate_std,
            Column.NoiseWhiteness.Z_SCORE: z_scores,
            Column.NoiseWhiteness.SURROGATE_P_VALUE: surrogate_p_values,
            Column.NoiseWhiteness.ANALYTIC_P_VALUE: analytic_p_values.to_numpy(),
        }
    )


def _build_correlation_dataframe(
    correlations: np.ndarray, null_correlations: np.ndarray, column_names: list[str]
) -> pd.DataFrame:
    """Assemble the lagged correlation profile with its surrogate confidence band."""
    lower_percentile, upper_percentile = SURROGATE_PERCENTILES
    lower_band = np.nanpercentile(null_correlations, lower_percentile, axis=0)
    upper_band = np.nanpercentile(null_correlations, upper_percentile, axis=0)

    lags = np.arange(correlations.shape[0])
    num_features = len(column_names)

    rows = []
    for i in range(num_features):
        for j in range(num_features):
            rows.append(
                pd.DataFrame(
                    {
                        Column.NoiseWhiteness.FEATURE_PAIR: f"{column_names[i]}_{column_names[j]}",
                        Column.NoiseWhiteness.LAG: lags,
                        Column.NoiseWhiteness.RHO: correlations[:, i, j],
                        Column.NoiseWhiteness.RHO_SURROGATE_LOWER: lower_band[:, i, j],
                        Column.NoiseWhiteness.RHO_SURROGATE_UPPER: upper_band[:, i, j],
                    }
                )
            )

    return pd.concat(rows, ignore_index=True)
