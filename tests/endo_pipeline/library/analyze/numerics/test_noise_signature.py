import numpy as np
import pytest

from endo_pipeline.library.analyze.numerics.noise_signature import (
    SUMMARY_STATISTICS,
    NoiseStatistic,
    compute_integrated_correlation_time,
    compute_lagged_covariances,
    compute_measurement_noise_fraction,
    compute_measurement_noise_variance,
    compute_noise_signature,
    compute_power_spectrum,
    compute_reliability_ratio,
    compute_spectral_exponent,
    normalize_lagged_covariances,
    permute_residuals_in_time,
)
from endo_pipeline.settings.column_names import ColumnName as Column

COLUMN_NAMES = ["feature_a", "feature_b"]


def make_white_residuals(
    n_patches: int = 40, n_timepoints: int = 64, n_dim: int = 2, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_patches, n_timepoints, n_dim))


def make_colored_residuals(
    n_patches: int = 40,
    n_timepoints: int = 64,
    n_dim: int = 2,
    correlation: float = 0.8,
    seed: int = 0,
) -> np.ndarray:
    """Build residuals from an AR(1) process with a known positive correlation."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(n_patches, n_timepoints, n_dim))
    residuals = np.empty_like(innovations)
    residuals[:, 0, :] = innovations[:, 0, :]
    for timepoint in range(1, n_timepoints):
        residuals[:, timepoint, :] = (
            correlation * residuals[:, timepoint - 1, :] + innovations[:, timepoint, :]
        )
    return residuals


def make_measurement_error_residuals(
    n_patches: int = 400,
    n_timepoints: int = 64,
    dynamic_step: float = 0.05,
    localization_error: float = 0.03,
    seed: int = 3,
) -> np.ndarray:
    """Difference a random walk that is observed with independent error."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=dynamic_step, size=(n_patches, n_timepoints + 1, 2))
    positions = np.cumsum(steps, axis=1)
    observed = positions + rng.normal(scale=localization_error, size=positions.shape)
    return observed[:, 1:, :] - observed[:, :-1, :]


def statistic_values(statistics):
    return statistics.set_index([Column.NoiseSignature.STATISTIC, Column.NoiseSignature.FEATURE])[
        Column.NoiseSignature.VALUE
    ]


def test_lagged_covariances_recover_instantaneous_covariance():
    residuals = make_white_residuals(n_patches=200, n_timepoints=64)
    covariances, counts = compute_lagged_covariances(residuals, max_lag=5)

    assert counts[0] == 200 * 64
    assert counts[3] == 200 * (64 - 3)
    # white noise has unit variance and no cross-feature covariance
    np.testing.assert_allclose(covariances[0], np.eye(2), atol=0.05)


def test_lagged_covariances_excludes_pairs_across_gaps():
    residuals = make_white_residuals(n_patches=3, n_timepoints=10)
    residuals[:, 4, :] = np.nan

    _, counts = compute_lagged_covariances(residuals, max_lag=1)

    # 9 timepoints remain per patch, and the gap removes two of the nine lag-1 pairs
    assert counts[0] == 3 * 9
    assert counts[1] == 3 * 7


def test_white_residuals_have_vanishing_lagged_correlations():
    residuals = make_white_residuals(n_patches=400, n_timepoints=64)
    covariances, _ = compute_lagged_covariances(residuals, max_lag=5)
    correlations = normalize_lagged_covariances(covariances)

    np.testing.assert_allclose(np.diag(correlations[0]), np.ones(2), atol=1e-12)
    assert np.abs(np.einsum("lii->li", correlations[1:])).max() < 0.05


def test_ar1_residuals_recover_known_lag_one_correlation():
    correlation = 0.8
    residuals = make_colored_residuals(n_patches=400, n_timepoints=128, correlation=correlation)
    covariances, _ = compute_lagged_covariances(residuals, max_lag=3)
    correlations = normalize_lagged_covariances(covariances)

    np.testing.assert_allclose(np.diag(correlations[1]), correlation, atol=0.05)
    np.testing.assert_allclose(np.diag(correlations[2]), correlation**2, atol=0.05)


def test_integrated_correlation_time_is_one_half_for_white_noise():
    residuals = make_white_residuals(n_patches=400, n_timepoints=64)
    covariances, _ = compute_lagged_covariances(residuals, max_lag=10)
    correlations = normalize_lagged_covariances(covariances)

    np.testing.assert_allclose(compute_integrated_correlation_time(correlations), 0.5, atol=0.1)


def test_integrated_correlation_time_grows_for_correlated_noise():
    residuals = make_colored_residuals(n_patches=400, n_timepoints=128, correlation=0.8)
    covariances, _ = compute_lagged_covariances(residuals, max_lag=20)
    correlations = normalize_lagged_covariances(covariances)

    assert (compute_integrated_correlation_time(correlations) > 2.0).all()


def test_integrated_correlation_time_truncates_on_negative_lag_one():
    residuals = make_measurement_error_residuals()
    covariances, _ = compute_lagged_covariances(residuals, max_lag=10)
    correlations = normalize_lagged_covariances(covariances)

    # anti-persistent noise carries no memory to accumulate
    np.testing.assert_allclose(compute_integrated_correlation_time(correlations), 0.5, atol=1e-12)


def test_spectral_exponent_is_zero_for_white_noise():
    residuals = make_white_residuals(n_patches=400, n_timepoints=64)
    frequencies, power_spectrum = compute_power_spectrum(residuals, segment_length=32)

    exponents, _ = compute_spectral_exponent(frequencies, power_spectrum)

    np.testing.assert_allclose(exponents, 0.0, atol=0.15)


def test_spectral_exponent_is_positive_for_persistent_noise():
    residuals = make_colored_residuals(n_patches=400, n_timepoints=128, correlation=0.8)
    frequencies, power_spectrum = compute_power_spectrum(residuals, segment_length=32)

    exponents, r_squared = compute_spectral_exponent(frequencies, power_spectrum)

    assert (exponents > 0.8).all()
    # an AR(1) spectrum is close to a power law over this band
    assert (r_squared > 0.9).all()


def test_spectral_exponent_is_negative_for_measurement_error():
    residuals = make_measurement_error_residuals()
    frequencies, power_spectrum = compute_power_spectrum(residuals, segment_length=32)

    exponents, _ = compute_spectral_exponent(frequencies, power_spectrum)

    # differenced observation error puts power at high frequency
    assert (exponents < -0.1).all()


def test_spectral_exponent_fit_quality_is_low_for_a_flat_spectrum():
    residuals = make_white_residuals(n_patches=400, n_timepoints=64)
    frequencies, power_spectrum = compute_power_spectrum(residuals, segment_length=32)

    _, r_squared = compute_spectral_exponent(frequencies, power_spectrum)

    # white noise has no trend for the power law to explain
    assert (r_squared < 0.5).all()


def test_measurement_noise_variance_recovers_known_error():
    localization_error = 0.03
    residuals = make_measurement_error_residuals(localization_error=localization_error)
    covariances, _ = compute_lagged_covariances(residuals, max_lag=2)
    correlations = normalize_lagged_covariances(covariances)

    noise_variance = compute_measurement_noise_variance(correlations, covariances)

    np.testing.assert_allclose(noise_variance, localization_error**2, rtol=0.1)


def test_measurement_noise_fraction_recovers_pure_measurement_error():
    # differencing a pure measurement error signal gives rho(1) = -0.5
    rng = np.random.default_rng(0)
    errors = rng.normal(size=(400, 65, 2))
    residuals = np.diff(errors, axis=1)

    covariances, _ = compute_lagged_covariances(residuals, max_lag=2)
    correlations = normalize_lagged_covariances(covariances)

    np.testing.assert_allclose(compute_measurement_noise_fraction(correlations), 1.0, atol=0.05)


def test_measurement_noise_estimates_are_nan_for_persistent_noise():
    residuals = make_colored_residuals(n_patches=100, n_timepoints=64, correlation=0.8)
    covariances, _ = compute_lagged_covariances(residuals, max_lag=2)
    correlations = normalize_lagged_covariances(covariances)

    assert np.isnan(compute_measurement_noise_fraction(correlations)).all()
    assert np.isnan(compute_measurement_noise_variance(correlations, covariances)).all()


def test_reliability_ratio_matches_errors_in_variables_definition():
    noise_variance = np.array([0.0025, 0.0025])
    feature_variance = np.array([1.0, 0.01])

    reliability = compute_reliability_ratio(noise_variance, feature_variance)

    np.testing.assert_allclose(reliability, [1 - 0.0025, 1 - 0.25])
    # a feature with a narrow spread suffers far more attenuation
    assert reliability[1] < reliability[0]


def test_reliability_ratio_is_nan_without_a_measurement_error_estimate():
    reliability = compute_reliability_ratio(np.array([np.nan, 0.01]), np.array([1.0, 1.0]))

    assert np.isnan(reliability[0])
    assert reliability[1] == pytest.approx(0.99)


def test_permutation_preserves_missing_data_mask_and_values():
    residuals = make_white_residuals(n_patches=5, n_timepoints=12)
    residuals[1, 3:6, :] = np.nan
    residuals[4, 0, :] = np.nan

    permuted = permute_residuals_in_time(residuals, np.random.default_rng(0))

    np.testing.assert_array_equal(np.isnan(permuted), np.isnan(residuals))
    for patch_index in range(residuals.shape[0]):
        original = residuals[patch_index][~np.isnan(residuals[patch_index]).any(axis=1)]
        shuffled = permuted[patch_index][~np.isnan(permuted[patch_index]).any(axis=1)]
        # the same residual vectors are present, only their time order changes
        np.testing.assert_allclose(np.sort(original, axis=0), np.sort(shuffled, axis=0), atol=1e-12)


def test_permutation_destroys_temporal_correlation():
    residuals = make_colored_residuals(n_patches=200, n_timepoints=64, correlation=0.8)
    permuted = permute_residuals_in_time(residuals, np.random.default_rng(0))

    def lag_one(array: np.ndarray) -> float:
        covariances, _ = compute_lagged_covariances(array, max_lag=1)
        return float(np.diag(normalize_lagged_covariances(covariances)[1]).max())

    # permutation keeps each patch's mean residual, so a small positive offset
    # survives; the temporal correlation itself is removed
    assert lag_one(permuted) < 0.2
    assert lag_one(permuted) < 0.3 * lag_one(residuals)


def test_noise_signature_does_not_flag_white_noise():
    residuals = make_white_residuals(n_patches=100, n_timepoints=64)

    result = compute_noise_signature(
        residuals, COLUMN_NAMES, max_lag=5, num_surrogates=50, segment_length=32
    )

    assert result is not None
    values = result.statistics.set_index(
        [Column.NoiseSignature.STATISTIC, Column.NoiseSignature.FEATURE]
    )
    for column_name in COLUMN_NAMES:
        for statistic in (
            NoiseStatistic.LAG_ONE_CORRELATION,
            NoiseStatistic.INTEGRATED_CORRELATION_TIME,
            NoiseStatistic.SPECTRAL_EXPONENT,
        ):
            row = values.loc[(statistic, column_name)]
            assert row[Column.NoiseSignature.P_VALUE_PERMUTATION] > 0.01


def test_noise_signature_flags_persistent_noise():
    residuals = make_colored_residuals(n_patches=100, n_timepoints=64, correlation=0.8)

    result = compute_noise_signature(
        residuals, COLUMN_NAMES, max_lag=5, num_surrogates=50, segment_length=32
    )

    assert result is not None
    values = result.statistics.set_index(
        [Column.NoiseSignature.STATISTIC, Column.NoiseSignature.FEATURE]
    )
    for column_name in COLUMN_NAMES:
        for statistic in (
            NoiseStatistic.LAG_ONE_CORRELATION,
            NoiseStatistic.SPECTRAL_EXPONENT,
        ):
            row = values.loc[(statistic, column_name)]
            assert row[Column.NoiseSignature.P_VALUE_PERMUTATION] <= 0.05
            assert abs(row[Column.NoiseSignature.Z_SCORE]) > 3.0


def test_noise_signature_separates_measurement_error_from_persistent_noise():
    localization_error = 0.03
    residuals = make_measurement_error_residuals(
        n_patches=200, localization_error=localization_error
    )

    result = compute_noise_signature(
        residuals, COLUMN_NAMES, max_lag=5, num_surrogates=50, segment_length=32
    )

    assert result is not None
    values = statistic_values(result.statistics)
    for column_name in COLUMN_NAMES:
        assert values[(NoiseStatistic.LAG_ONE_CORRELATION, column_name)] < 0
        assert values[(NoiseStatistic.SPECTRAL_EXPONENT, column_name)] < 0
        assert values[(NoiseStatistic.INTEGRATED_CORRELATION_TIME, column_name)] == pytest.approx(
            0.5
        )
        assert values[(NoiseStatistic.MEASUREMENT_NOISE_VARIANCE, column_name)] == pytest.approx(
            localization_error**2, rel=0.15
        )


def test_noise_signature_reports_reliability_when_feature_variance_given():
    residuals = make_measurement_error_residuals(n_patches=200, localization_error=0.03)
    feature_variance = np.array([1.0, 1.0])

    result = compute_noise_signature(
        residuals,
        COLUMN_NAMES,
        max_lag=5,
        num_surrogates=10,
        segment_length=32,
        feature_variance=feature_variance,
    )

    assert result is not None
    values = statistic_values(result.statistics)
    for column_name in COLUMN_NAMES:
        noise_variance = values[(NoiseStatistic.MEASUREMENT_NOISE_VARIANCE, column_name)]
        assert values[(NoiseStatistic.RELIABILITY_RATIO, column_name)] == pytest.approx(
            1 - noise_variance
        )


def test_noise_signature_reliability_is_nan_without_feature_variance():
    residuals = make_measurement_error_residuals(n_patches=100)

    result = compute_noise_signature(
        residuals, COLUMN_NAMES, max_lag=5, num_surrogates=5, segment_length=32
    )

    assert result is not None
    values = statistic_values(result.statistics)
    assert np.isnan(values[(NoiseStatistic.RELIABILITY_RATIO, COLUMN_NAMES[0])])


def test_noise_signature_result_shapes_and_metadata():
    residuals = make_white_residuals(n_patches=20, n_timepoints=64)

    result = compute_noise_signature(
        residuals,
        COLUMN_NAMES,
        max_lag=5,
        num_surrogates=10,
        segment_length=32,
        metadata_dict={Column.DATASET: "test_dataset"},
    )

    assert result is not None
    assert result.lags.shape == (6,)
    assert result.correlations.shape == (6, 2, 2)
    assert result.correlation_bounds.shape == (2, 6, 2)
    assert result.frequencies.shape[0] == result.power_spectrum.shape[0]
    assert result.power_spectrum.shape[1] == 2
    assert (result.statistics[Column.DATASET] == "test_dataset").all()


def test_noise_signature_handles_gapped_residuals():
    rng = np.random.default_rng(1)
    residuals = make_white_residuals(n_patches=30, n_timepoints=64)
    residuals[rng.random((30, 64)) < 0.2] = np.nan

    result = compute_noise_signature(
        residuals, COLUMN_NAMES, max_lag=5, num_surrogates=10, segment_length=32
    )

    assert result is not None
    assert np.isfinite(result.correlations[0]).all()


@pytest.mark.parametrize(
    "residuals",
    [np.zeros((5, 10)), np.zeros((5, 10, 3))],
)
def test_noise_signature_returns_none_for_mismatched_input(residuals):
    assert compute_noise_signature(residuals, COLUMN_NAMES) is None


def test_noise_signature_returns_none_for_single_timepoint():
    assert compute_noise_signature(np.zeros((5, 1, 2)), COLUMN_NAMES) is None


def test_summary_statistics_are_all_present_in_the_output():
    residuals = make_measurement_error_residuals(n_patches=100)

    result = compute_noise_signature(
        residuals,
        COLUMN_NAMES,
        max_lag=5,
        num_surrogates=5,
        segment_length=32,
        feature_variance=np.array([1.0, 1.0]),
    )

    assert result is not None
    reported = set(result.statistics[Column.NoiseSignature.STATISTIC])
    assert {str(statistic) for statistic in SUMMARY_STATISTICS} <= reported
