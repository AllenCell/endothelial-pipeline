import numpy as np
import pandas as pd
import pytest
from scipy.stats import expon, kstest

from endo_pipeline.library.analyze.numerics.correlations import (
    MIN_VALID_SAMPLES_FOR_PERIODOGRAM,
    NoiseCorrelationCentering,
    compute_lombscargle_periodogram,
    compute_periodogram_dataframe,
)
from endo_pipeline.library.analyze.numerics.forward_difference import (
    compute_forward_differences_along_trajectory,
)
from endo_pipeline.settings.column_names import ColumnName as Column


def expected_power_spectrum(
    frequencies: np.ndarray, d_coeff: float, sigma_2: float, dt: float
) -> np.ndarray:
    """Eq. (13) of Vestergaard et al. (2014), Phys. Rev. E 89, 022726, assuming no motion blur (R=0)."""
    return 2 * d_coeff * dt**2 + 2 * sigma_2 * dt * (1 - np.cos(2 * np.pi * frequencies * dt))


def simulate_white_noise_residuals(
    rng: np.random.Generator,
    n_patches: int,
    num_samples: int,
    d_coeff: float,
    sigma_2: float,
    dt: float,
) -> np.ndarray:
    """Simulate single-feature white-noise residuals matching Eq. (2) of the reference, with R=0."""
    variance = 2 * d_coeff * dt + 2 * sigma_2
    return rng.normal(scale=np.sqrt(variance), size=(n_patches, num_samples, 1))


def test_lombscargle_periodogram_matches_expected_power_spectrum_for_complete_data():
    rng = np.random.default_rng(0)
    dt, d_coeff, sigma_2 = 0.5, 2.0, 0.5
    num_samples, n_patches = 2000, 1
    residuals = simulate_white_noise_residuals(rng, n_patches, num_samples, d_coeff, sigma_2, dt)
    timepoints_array = np.arange(num_samples)

    periodogram, frequencies = compute_lombscargle_periodogram(residuals, timepoints_array, dt)
    expected = expected_power_spectrum(frequencies, d_coeff, sigma_2, dt)

    assert periodogram.mean() == pytest.approx(expected.mean(), rel=0.1)


def test_lombscargle_periodogram_normalized_values_follow_exponential_null_distribution():
    rng = np.random.default_rng(1)
    dt, d_coeff, sigma_2 = 0.5, 2.0, 0.5
    num_samples, n_patches = 100, 30
    residuals = simulate_white_noise_residuals(rng, n_patches, num_samples, d_coeff, sigma_2, dt)
    timepoints_array = np.arange(num_samples)

    periodogram, frequencies = compute_lombscargle_periodogram(residuals, timepoints_array, dt)
    expected = expected_power_spectrum(frequencies, d_coeff, sigma_2, dt)
    normalized = (periodogram[:, :, 0] / expected[np.newaxis, :]).ravel()

    assert normalized.mean() == pytest.approx(1.0, rel=0.3)
    _, p_value = kstest(normalized, expon(scale=1).cdf)
    assert p_value > 0.01


def test_lombscargle_periodogram_is_unbiased_with_missing_timepoints():
    # Zero-filling missing timepoints biases the periodogram toward lower
    # power roughly in proportion to the fraction of missing data; this
    # guards against that regression.
    rng = np.random.default_rng(2)
    dt, d_coeff, sigma_2 = 0.5, 2.0, 0.5
    num_samples, n_patches = 200, 200
    residuals = simulate_white_noise_residuals(rng, n_patches, num_samples, d_coeff, sigma_2, dt)
    timepoints_array = np.arange(num_samples)

    # simulate per-patch track start/end gaps, as from finite cell-tracking duration
    for patch_index in range(n_patches):
        start = rng.integers(0, num_samples // 2)
        length = rng.integers(num_samples // 4, num_samples - start)
        end = min(start + length, num_samples)
        residuals[patch_index, :start, 0] = np.nan
        residuals[patch_index, end:, 0] = np.nan
    valid_fraction = np.isfinite(residuals).mean()
    assert valid_fraction < 0.8, "test setup should produce substantial missingness"

    periodogram, frequencies = compute_lombscargle_periodogram(residuals, timepoints_array, dt)
    expected = expected_power_spectrum(frequencies, d_coeff, sigma_2, dt)
    normalized = (periodogram[:, :, 0] / expected[np.newaxis, :]).ravel()
    normalized = normalized[np.isfinite(normalized)]

    assert normalized.mean() == pytest.approx(1.0, rel=0.3)


def test_lombscargle_periodogram_is_nan_for_patches_with_too_few_valid_samples():
    num_samples = 50
    residuals = np.zeros((2, num_samples, 1))
    # only 1 valid timepoint, below MIN_VALID_SAMPLES_FOR_PERIODOGRAM
    residuals[0, 1:, 0] = np.nan
    assert np.isfinite(residuals[0, :, 0]).sum() < MIN_VALID_SAMPLES_FOR_PERIODOGRAM
    timepoints_array = np.arange(num_samples)

    periodogram, _ = compute_lombscargle_periodogram(residuals, timepoints_array, dt=1.0)

    assert np.all(np.isnan(periodogram[0]))
    assert np.all(np.isfinite(periodogram[1]))


def test_compute_periodogram_dataframe_end_to_end_with_missing_timepoints():
    rng = np.random.default_rng(3)
    column_names = ["feature"]
    timepoints_array = np.arange(20)

    traj_list, d_traj_list = [], []
    for patch_timepoints in (timepoints_array, timepoints_array[5:18]):
        values = rng.normal(size=len(patch_timepoints))
        df = pd.DataFrame({Column.TIMEPOINT: patch_timepoints, "feature": values})
        traj, d_traj = compute_forward_differences_along_trajectory(df, column_names)
        traj_list.append(traj)
        d_traj_list.append(d_traj)

    periodogram_dataframe = compute_periodogram_dataframe(
        traj_list=traj_list,
        d_traj_list=d_traj_list,
        column_names=column_names,
        timepoints_array=timepoints_array,
        vector_field=lambda x: np.zeros_like(x),
        centering_method=NoiseCorrelationCentering.DRIFT_SUBTRACTED,
        dt=1.0,
    )

    expected_columns = {
        Column.Periodogram.FEATURE,
        Column.Periodogram.PATCH_INDEX,
        Column.Periodogram.FREQUENCY,
        Column.Periodogram.POWER,
    }
    assert expected_columns.issubset(periodogram_dataframe.columns)
    assert not periodogram_dataframe[Column.Periodogram.POWER].isna().any()
    assert set(periodogram_dataframe[Column.Periodogram.PATCH_INDEX].unique()) == {0, 1}
