"""Methods for computing autocorrelation and cross-correlation functions from time series data."""

import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.cross_correlations import MAX_LAG_INTEGRATE, NUM_TIMEPOINT_FRAC
from endo_pipeline.settings.dynamics_workflows import POLAR_ANGLE_PERIOD, RESCALE_THETA

logger = logging.getLogger(__name__)


def cross_correlation_function(
    data_feat1: np.ndarray, data_feat2: np.ndarray, lag_cutoff_fraction: int = NUM_TIMEPOINT_FRAC
) -> np.ndarray:
    """Get the normalized cross-correlation function (CCF) between two features.

    The input data arrays are expected to be of shape (num_samples,
    num_timepoints). That is, the data are assumed to be {num_samples} iid
    sample time series, each sampled at the same num_timepoints.

    The cross correlation function for each sample is computed using the
    convolution theorem, which states that the CCF is the inverse Fourier
    transform (using the fast Fourier transform, or FFT) of the cross power
    spectrum of the two signals. That is, it is equal to the inverse FFT of
    `X1^{*}(f) * X2(f)` where `X1` and `X2` are the FFTs of the two signals.

    The CCF is normalized by the product of the standard deviations of the two
    signals to get the scaled CCF.

    The resulting scaled CCF is then shifted so that zero lag is in the center
    of the array, and only the middle portion of the CCF is returned
    (corresponding to lags going from - to +
    `num_timepoints//lag_cutoff_fraction`) to get the actual CCF of the unpadded
    signal.

    Finally, the CCF is averaged over the num_samples trajectories to get the
    average CCF across the population of samples.

    Parameters
    ----------
    data_feat1
        Array of shape (num_samples, num_timepoints) containing time series data
        for the first feature for the CCF.
    data_feat2
        Array of shape (num_samples, num_timepoints) containing time series data
        for the second feature for the CCF.
    lag_cutoff_fraction
        Fraction of num_timepoints to use as cutoff for lags in the returned
        CCF.

    Returns
    -------
    :
        Array of shape (num_lags,) containing the average scaled CCF across the population of samples,
        where num_lags is equal to `2 * num_timepoints // lag_cutoff_fraction + 1`.

    """
    num_traj = data_feat1.shape[0]
    num_timepoints = data_feat1.shape[1]

    # Get nearest power of 2 greater than 2*num_timepoints-1.
    # This is to pass into np.fft.fft to pad the signal with
    # zeros for efficient FFT computation (Cooley-Tukey algorithm)
    num_pad = 2 ** int(np.ceil(np.log2(2 * num_timepoints - 1)))

    for traj_index in range(num_traj):
        # Center data by subtracting mean, get standard deviation
        # for normalization of CCF.
        # FFT cannot handle NaNs, so we replace them with zeros after
        # centering/mean subtraction.
        data_mean1 = np.nanmean(data_feat1[traj_index])
        data_stdev1 = np.nanstd(data_feat1[traj_index], ddof=1)
        x_t_i_ctr = data_feat1[traj_index] - data_mean1
        x_t_i_ctr = np.nan_to_num(x_t_i_ctr, nan=0.0)

        data_mean2 = np.nanmean(data_feat2[traj_index])
        data_stdev2 = np.nanstd(data_feat2[traj_index], ddof=1)
        x_t_j_ctr = data_feat2[traj_index] - data_mean2
        x_t_j_ctr = np.nan_to_num(x_t_j_ctr, nan=0.0)

        # Get the FFT of the centered data, padding with zeros to length num_pad.
        cf_1 = np.fft.fft(x_t_i_ctr, n=num_pad)
        cf_2 = np.fft.fft(x_t_j_ctr, n=num_pad)
        # Compute the cross power spectrum of the padded signals (normalized by num_timepoints).
        sf = cf_1.conjugate() * cf_2 / num_timepoints

        # Compute the inverse FFT of the power spectrum to get the CCF,
        # normalizing by product of standard deviations (definition of scaled CCF)
        corr_unshifted = np.fft.ifft(sf).real / (data_stdev1 * data_stdev2)

        # Shift the CCF so that zero lag is in the center of the array and
        # extract the middle portion of the CCF corresponding to lags from - to
        # + num_timepoints//lag_cutoff_fraction.
        corr_shifted = np.fft.fftshift(corr_unshifted)
        max_lag = num_timepoints // lag_cutoff_fraction
        index_lb = num_pad // 2 - max_lag
        index_ub = num_pad // 2 + max_lag + 1
        corr = corr_shifted[index_lb:index_ub]

        # Running sum over trajectories to get average.
        if traj_index == 0:
            corr_sum = corr
        else:
            corr_sum = corr_sum + corr

        if np.isnan(corr_sum).any():
            logger.warning(
                "NaN values found in CCF for trajectory index [ %s ]. "
                "This may be due to zero standard deviation in one of the signals for this trajectory.",
                traj_index,
            )
            break

    # Return average over number of trajectories.
    return corr_sum / num_traj


def autocorrelation_function(
    data: np.ndarray, component_index: int, lag_cutoff_fraction: int = NUM_TIMEPOINT_FRAC
) -> np.ndarray:
    """Get the normalized autocorrelation function (ACF) for a specific component.

    Wrapper for `cross_correlation_function`, using the fact that the ACF is just
    the CCF of a signal with itself.

    Parameters
    ----------
    data
        Array of shape (num_samples, num_timepoints, num_components) containing
        time series data for the feature of interest.
    component_index
        Index of the component for which to compute the ACF.
    lag_cutoff_fraction
        Fraction of num_timepoints to use as cutoff for lags in the returned ACF.

    Returns
    -------
    :
        Array of shape (num_lags,) containing the average scaled ACF across the
        population of samples, where num_lags is equal to `2 * num_timepoints // lag_cutoff_fraction + 1`.

    """
    # Extract the specified component from the data array.
    x_t_j = data[..., component_index]

    # Pass to cross_correlation_function with itself to get ACF.
    return cross_correlation_function(x_t_j, x_t_j, lag_cutoff_fraction=lag_cutoff_fraction)


def fit_exp_decay_and_get_relaxation_timescale(
    acf: np.ndarray,
    lags: np.ndarray,
    exp_decay_func: Literal["exponential_decay", "double_exponential_decay"],
    maxfev: int = 10000,
) -> tuple[np.ndarray, float]:
    """Fit exponential decay to ACF and return fit parameters and relaxation timescale."""
    # check to make sure valid function is provided
    if exp_decay_func not in ["exponential_decay", "double_exponential_decay"]:
        logger.error(
            "Invalid exp_decay_func provided: [ %s ]. "
            "Must be 'exponential_decay' or 'double_exponential_decay'.",
            exp_decay_func,
        )
        raise ValueError(
            "Invalid exp_decay_func provided to _fit_exp_decay_and_get_relaxation_timescale."
        )

    # get indices where both lags and acf are finite, as required for input to curve_fit function
    valid_indices = np.isfinite(acf) & np.isfinite(lags)

    if exp_decay_func == "exponential_decay":
        p0 = [0.5, 0.5, 0.5]  # initial guess for a, b, and c
        exp_fit, _ = curve_fit(
            exponential_decay, lags[valid_indices], acf[valid_indices], maxfev=maxfev, p0=p0
        )
        relaxation_time = 1 / exp_fit[1]
    else:
        p0 = [0.5, 0.5, 0.5, 0.5, 0.5]  # initial guess for a1, b1, a2, b2, and c
        exp_fit, _ = curve_fit(
            double_exponential_decay,
            lags[valid_indices],
            acf[valid_indices],
            maxfev=maxfev,
            p0=p0,
        )
        # choose the relaxation time corresponding to the larger weight
        which_weight_is_larger = np.argmax(np.abs(exp_fit[[0, 2]]))
        relaxation_time = 1 / exp_fit[[1, 3][which_weight_is_larger]]

    return exp_fit, relaxation_time


def cross_correlation_difference_norm(
    delta_ccf: np.ndarray, max_lag_integrate: int = MAX_LAG_INTEGRATE
) -> np.ndarray:
    """Compute the L2 norm of the difference between positive and negative lags of the CCF."""
    delta_ccf_over_interval = delta_ccf[:max_lag_integrate]
    integral_norm = np.sqrt(np.trapz(delta_ccf_over_interval**2, axis=0, dx=1) / max_lag_integrate)
    return integral_norm


def compute_autocorrelation_and_relaxation_for_one_bootstrap_sample(
    data: np.ndarray,
    component_index: int,
    lags: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute the ACF and relaxation timescale for one bootstrap sample."""
    # Random sampling with replacement to generate bootstrap samples
    num_traj = data.shape[0]
    inds = np.random.choice(num_traj, num_traj, replace=True)
    data_resampled = data[inds]

    # Calculate cross-correlation for each iteration of resampling and append to list
    acf = autocorrelation_function(data_resampled, component_index)

    # Fit exponential decay to ACF and get relaxation timescale
    index_positive = lags > 0
    positive_lags = lags[index_positive]
    positive_lags_as_hours = 5 * positive_lags / 60  # convert from frames (5 minutes) to hours
    acf_positive_lags = acf[index_positive]

    _, relaxation_time = fit_exp_decay_and_get_relaxation_timescale(
        acf_positive_lags, positive_lags_as_hours, exp_decay_func="exponential_decay"
    )

    return acf, relaxation_time


def bootstrap_autocorrelation_confidence_intervals(
    data: np.ndarray,
    component_index: int,
    lags: np.ndarray,
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
    max_cores: int | None = None,
) -> dict[str, tuple]:
    """Bootstrap the normalized autocorrelation function (ACF) computed from finite data.

    The ACF is computed for a specific vector component of an ensemble of stationary,
    vector-valued time series data.

    The input data array is expected to be of shape (num_samples, num_timepoints, num_components).
    That is, the data are assumed to be {num_samples} iid samples, each sampled at
    the same num_timepoints and having the same num_components.

    This function resamples the data with replacement to generate bootstrap samples and
    computes the ACF for each bootstrap sample. It then calculates the confidence interval
    for the ACF and related quantities based on the bootstrap samples.

    Parameters
    ----------
    data
        Array of shape (num_samples, num_timepoints, num_components) containing time series data.
    component_index
        Index of the vector component for which to compute the ACF.
    lags
        Array of lag values corresponding to the ACF.
    n_bootstraps
        Number of bootstrap samples to generate for the ACF.
    confidence_level
        Confidence interval level (e.g., 95%) to report for the ACF.
    max_cores
        Maximum number of CPU cores to use for parallel processing of bootstrap
        samples. If None, will use all available cores.

    Returns
    -------
    :
        Dictionary containing lower and upper bounds of the confidence intervals for:
        - autocorrelation: ACF(tau)
        - relaxation_timescale: Decay coefficient from exponential fit to ACF.

    """
    # Bootstrap the ACF using resampling with replacement
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        results = executor.map(
            compute_autocorrelation_and_relaxation_for_one_bootstrap_sample,
            [data] * n_bootstraps,
            [component_index] * n_bootstraps,
            [lags] * n_bootstraps,
        )
        bootstrap_autocorrelations, bootstrap_relaxation_timescales = zip(*results, strict=False)

    # Calculate the lower and upper bounds of the confidence interval
    percentile = (1 - confidence_level) / 2
    acf_lower_bound = np.percentile(bootstrap_autocorrelations, 100 * percentile, axis=0)
    acf_upper_bound = np.percentile(bootstrap_autocorrelations, 100 * (1 - percentile), axis=0)
    relaxation_time_lower_bound = np.percentile(
        bootstrap_relaxation_timescales, 100 * percentile, axis=0
    )
    relaxation_time_upper_bound = np.percentile(
        bootstrap_relaxation_timescales, 100 * (1 - percentile), axis=0
    )
    confidence_interval_bounds = {
        "autocorrelation": (acf_lower_bound, acf_upper_bound),
        "relaxation_timescale": (relaxation_time_lower_bound, relaxation_time_upper_bound),
    }

    return confidence_interval_bounds


def compute_crosscorrelation_and_delta_crosscorrelation_for_one_bootstrap_sample(
    data_feat1: np.ndarray,
    data_feat2: np.ndarray,
    max_lag_integrate: int = MAX_LAG_INTEGRATE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the CCF, delta CCF, and delta CCF integral for one bootstrap sample."""

    if data_feat1.shape[0] != data_feat2.shape[0]:
        logger.error(
            "Input data arrays must have the same number of trajectories. Got [ %s ] and [ %s ].",
            data_feat1.shape[0],
            data_feat2.shape[0],
        )
        raise ValueError("Input data arrays must have the same number of trajectories.")

    # Random sampling with replacement to generate bootstrap samples
    num_traj = data_feat1.shape[0]
    inds = np.random.choice(num_traj, num_traj, replace=True)
    data_feat1_resampled = data_feat1[inds]
    data_feat2_resampled = data_feat2[inds]

    # Calculate cross-correlation for resampled data
    ccf = cross_correlation_function(data_feat1_resampled, data_feat2_resampled)

    # Calculate delta CCF and delta CCF integral for resampled data
    num_lags = len(ccf)
    delta_ccf = ccf[1 + num_lags // 2 :] - ccf[: num_lags // 2]
    delta_ccf_integral = cross_correlation_difference_norm(delta_ccf, max_lag_integrate)

    return ccf, delta_ccf, delta_ccf_integral


def bootstrap_cross_correlation_confidence_intervals(
    data_feat1: np.ndarray,
    data_feat2: np.ndarray,
    n_bootstraps: int = 200,
    max_lag_integrate: int = MAX_LAG_INTEGRATE,
    confidence_level: float = 0.95,
    max_cores: int | None = None,
) -> dict[str, tuple]:
    """Bootstrap the normalized cross-correlation function (CCF) computed from finite data.

    The CCF is computed between between vector components of an ensemble of stationary,
    vector-valued time series data.

    The input two input data arrays are each expected to be of shape (num_samples, num_timepoints).
    That is, the data are assumed to be {num_samples} iid samples, each sampled at
    the same num_timepoints.

    This function resamples the data with replacement to generate bootstrap samples and
    computes the CCF for each bootstrap sample. It then calculates the confidence interval
    for the CCF and related quantities based on the bootstrap samples.

    Parameters
    ----------
    data_feat1
        Array of shape (num_samples, num_timepoints) containing time series data for the
        first vector component for the CCF.
    data_feat2
        Array of shape (num_samples, num_timepoints) containing time series data for the
        second vector component for the CCF.
    max_lag_integrate
        Upper bound of integration for the delta CCF integral calculation.
    n_bootstraps
        Number of bootstrap samples to generate for the CCF.
    confidence_level
        Confidence interval level (e.g., 95%) to report for the CCF.
    max_cores
        Maximum number of CPU cores to use for parallel processing of bootstrap
        samples. If None, will use all available cores.

    Returns
    -------
    :
        Dictionary containing lower and upper bounds of the confidence intervals for:
        - cross_correlation: CCF(tau)
        - delta_cross_correlation: |CCF(tau>0) - CCF(tau<0)|
        - delta_cross_correlation_integral: Integral of |CCF(tau>0) - CCF(tau<0)| over
          the first {max_lag_integrate} lags.

    """
    # Bootstrap the CCF using resampling with replacement
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        results = executor.map(
            compute_crosscorrelation_and_delta_crosscorrelation_for_one_bootstrap_sample,
            [data_feat1] * n_bootstraps,
            [data_feat2] * n_bootstraps,
            [max_lag_integrate] * n_bootstraps,
        )
        (
            bootstrap_correlations,
            bootstrap_correlation_diffs,
            bootstrap_correlation_diff_integrals,
        ) = zip(*results, strict=False)

    # Calculate the lower and upper bounds of the confidence interval
    percentile = (1 - confidence_level) / 2
    ccf_lower_bound = np.percentile(bootstrap_correlations, 100 * percentile, axis=0)
    ccf_upper_bound = np.percentile(bootstrap_correlations, 100 * (1 - percentile), axis=0)
    delta_ccf_lower_bound = np.percentile(bootstrap_correlation_diffs, 100 * percentile, axis=0)
    delta_ccf_upper_bound = np.percentile(
        bootstrap_correlation_diffs, 100 * (1 - percentile), axis=0
    )
    delta_ccf_integral_lower_bound = np.percentile(
        bootstrap_correlation_diff_integrals, 100 * percentile, axis=0
    )
    delta_ccf_integral_upper_bound = np.percentile(
        bootstrap_correlation_diff_integrals, 100 * (1 - percentile), axis=0
    )

    confidence_interval_bounds = {
        "cross_correlation": (ccf_lower_bound, ccf_upper_bound),
        "delta_cross_correlation": (delta_ccf_lower_bound, delta_ccf_upper_bound),
        "delta_cross_correlation_integral": (
            delta_ccf_integral_lower_bound,
            delta_ccf_integral_upper_bound,
        ),
    }

    return confidence_interval_bounds


def compute_autocorrelation_dataframe(
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    metadata_dict: dict[str, str | float] | None = None,
) -> pd.DataFrame:
    """
    Compute autocorrelations for specified features, with bootstrap confidence
    intervals.

    For each trajectory in the dataframe (as indicated by `Column.CROP_INDEX`),
    this method computes the autocorrelation function (ACF) for each feature
    specified in column_names. It then saves the mean ACF across trajectories,
    as well as the `lower_percentile` and `upper_percentile` for the ACF at each
    lag, in a dataframe with columns for the dataset, crop index, lag, feature
    name, mean ACF, and ACF confidence interval bounds.

    Parameters
    ----------
    dataframe
        DataFrame containing the time series data for one dataset, with columns
        specified in column_names.
    column_names
        List of column names corresponding to the features for which to compute
        autocorrelations.
    lower_percentile
        Lower percentile to compute for the ACF at each lag.
    upper_percentile
        Upper percentile to compute for the ACF at each lag.

    """
    # check that required columns are present in the dataframe
    required_columns = [
        *column_names,
        Column.CROP_INDEX,
        Column.DATASET,
    ]
    check_required_columns_in_dataframe(dataframe, required_columns)

    # unwrap angles if polar_angle is in feat_cols
    if Column.DiffAEData.POLAR_ANGLE in column_names:
        for _, df_crop in dataframe.groupby(Column.CROP_INDEX):
            dataframe.loc[df_crop.index, Column.DiffAEData.POLAR_ANGLE] = np.unwrap(
                df_crop[Column.DiffAEData.POLAR_ANGLE], period=POLAR_ANGLE_PERIOD
            )

    # get feature data, filling missing timepoints with NaNs to ensure proper
    # alignment for correlation calculations
    t_min = dataframe[Column.TIMEPOINT].min()
    t_max = dataframe[Column.TIMEPOINT].max()
    all_timepoints = np.arange(t_min, t_max + 1)

    # fill missing timepoints with NaN values for each crop to ensure
    # consistent time axis across crops when computing population
    # variance and cumulative variance per crop, which require a 2D
    # array of shape (num_crops, num_timepoints)
    data_filled_list = []
    for _, data_crop in dataframe.groupby(Column.CROP_INDEX):
        # sort by timepoint to ensure correct order before reindexing
        data_crop = data_crop.sort_values(by=Column.TIMEPOINT)

        # reindex dataframe to include all timepoints in full range
        data_crop_filled = data_crop.set_index(Column.TIMEPOINT).reindex(all_timepoints)

        # reset index to restore timepoint column
        data_crop_filled = data_crop_filled.reset_index()

        # append to list
        data_filled_list.append(data_crop_filled)

    dataframe_filled = pd.concat(data_filled_list, ignore_index=True)

    num_feats = len(column_names)
    num_timepoints = len(all_timepoints)
    # make sure lags are symmetric around zero
    max_lags = num_timepoints // NUM_TIMEPOINT_FRAC
    lags = np.arange(-max_lags, max_lags + 1)
    num_lags = len(lags)

    # dataframe as array of shape (num_crops, num_timepoints, num_feats) for the
    # current feature, with missing timepoints filled with NaNs
    feats = dataframe_filled[column_names].to_numpy().reshape(-1, num_timepoints, num_feats)
    num_crops = feats.shape[0]
    acf_per_crop = np.zeros((num_crops, num_lags, num_feats))
    acf_dataframe_list = []
    for i in range(num_feats):
        for crop_index in range(num_crops):
            acf_per_crop[crop_index, :, i] = autocorrelation_function(feats, i)
        acf_mean = acf_per_crop[:, :, i].mean(axis=0)
        acf_lower_bound = np.percentile(acf_per_crop[:, :, i], lower_percentile, axis=0)
        acf_upper_bound = np.percentile(acf_per_crop[:, :, i], upper_percentile, axis=0)
        acf_dataframe_list.append(
            pd.DataFrame(
                {
                    Column.AutoCorrelation.FEATURE: column_names[i],
                    Column.AutoCorrelation.LAG: lags,
                    Column.AutoCorrelation.ACF_MEAN: acf_mean,
                    Column.AutoCorrelation.ACF_LOWER_PERCENTILE: acf_lower_bound,
                    Column.AutoCorrelation.ACF_UPPER_PERCENTILE: acf_upper_bound,
                }
            )
        )

    acf_dataframe = pd.concat(acf_dataframe_list, ignore_index=True)

    # add specified metadata columns to the dataframe (e.g. dataset name, shear
    # stress)
    if metadata_dict is not None:
        for key in metadata_dict:
            acf_dataframe[key] = metadata_dict[key]

    return acf_dataframe


def compute_correlations_for_one_dataset(
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    bootstrap_samples: int | None = None,
    max_lag_integrate: int = MAX_LAG_INTEGRATE,
    rescale_polar_angle: bool = RESCALE_THETA,
    max_cores: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute cross-correlation and autocorrelation for features from one dataset.

    Parameters
    ----------
    dataframe
        DataFrame containing the data for one dataset, with columns specified in
        column_names.
    column_names
        List of column names corresponding to the features for which to compute
        correlations.
    correlation_dict
        Dictionary to store the computed correlations, which will be updated and
        returned.
    bootstrap_samples
        Number of bootstrap samples to use for calculating confidence intervals.
        If None, no confidence intervals will be calculated.
    max_lag_integrate
        Maximum lag to integrate over for the forward minus backward CCF
        integral calculation.
    rescale_polar_angle
        Whether the polar angle variable has been rescaled from [-pi,pi] to
        [0,pi] (used to set the polar angle period for unwrapping).
    max_cores
        Maximum number of CPU cores to use for parallel processing of bootstrap
        samples. If None, will use all available cores.

    Returns
    -------
    :
        Updated correlation_dict with computed correlations for the dataset.

    """
    # check that required columns are present in the dataframe
    required_columns = [
        *column_names,
        Column.CROP_INDEX,
        Column.DATASET,
    ]
    check_required_columns_in_dataframe(dataframe, required_columns)

    # unwrap angles if polar_angle is in feat_cols
    if Column.DiffAEData.POLAR_ANGLE in column_names:
        polar_angle_period = POLAR_ANGLE_PERIOD if rescale_polar_angle else 2 * np.pi
        for _, df_crop in dataframe.groupby(Column.CROP_INDEX):
            dataframe.loc[df_crop.index, Column.DiffAEData.POLAR_ANGLE] = np.unwrap(
                df_crop[Column.DiffAEData.POLAR_ANGLE], period=polar_angle_period
            )

    # get feature data, filling missing timepoints with NaNs to ensure proper
    # alignment for correlation calculations
    t_min = dataframe[Column.TIMEPOINT].min()
    t_max = dataframe[Column.TIMEPOINT].max()
    all_timepoints = np.arange(t_min, t_max + 1)

    # fill missing timepoints with NaN values for each crop to ensure
    # consistent time axis across crops when computing population
    # variance and cumulative variance per crop, which require a 2D
    # array of shape (num_crops, num_timepoints)
    data_filled_list = []
    for _, data_crop in dataframe.groupby(Column.CROP_INDEX):
        # sort by timepoint to ensure correct order before reindexing
        data_crop = data_crop.sort_values(by=Column.TIMEPOINT)

        # reindex dataframe to include all timepoints in full range
        data_crop_filled = data_crop.set_index(Column.TIMEPOINT).reindex(all_timepoints)

        # reset index to restore timepoint column
        data_crop_filled = data_crop_filled.reset_index()

        # append to list
        data_filled_list.append(data_crop_filled)

    dataframe_filled = pd.concat(data_filled_list, ignore_index=True)

    num_feats = len(column_names)
    num_timepoints = len(all_timepoints)
    # make sure lags are symmetric around zero
    max_lags = num_timepoints // NUM_TIMEPOINT_FRAC
    lags = np.arange(-max_lags, max_lags + 1)

    num_lags = len(lags)
    # autocorrelation
    acf = np.zeros((num_lags, num_feats))
    acf_lb = np.zeros((num_lags, num_feats))
    acf_ub = np.zeros((num_lags, num_feats))
    relaxation_timescale = np.zeros(num_feats)
    relaxation_timescale_lb = np.zeros(num_feats)
    relaxation_timescale_ub = np.zeros(num_feats)
    # dataframe as array of shape (num_crops, num_timepoints, num_feats) for the
    # current feature, with missing timepoints filled with NaNs
    feats = dataframe_filled[column_names].to_numpy().reshape(-1, num_timepoints, num_feats)
    num_crops = feats.shape[0]
    acf_per_crop = np.zeros((num_crops, num_lags, num_feats))
    relaxation_timescale_per_crop = np.zeros((num_crops, num_feats))
    for i in range(num_feats):
        acf[:, i] = autocorrelation_function(feats, i)

        # Fit exponential decay to ACF and get relaxation timescale
        index_positive = lags > 0
        positive_lags = lags[index_positive]
        positive_lags_as_hours = 5 * positive_lags / 60  # convert from frames (5 minutes) to hours
        acf_positive_lags = acf[:, i][index_positive]

        _, relaxation_time = fit_exp_decay_and_get_relaxation_timescale(
            acf_positive_lags, positive_lags_as_hours, exp_decay_func="exponential_decay"
        )
        relaxation_timescale[i] = relaxation_time
        if bootstrap_samples is not None:
            # calculate bootstrap confidence intervals for ACF and relaxation timescale
            confidence_intervals = bootstrap_autocorrelation_confidence_intervals(
                feats, i, lags, n_bootstraps=bootstrap_samples, max_cores=max_cores
            )
            acf_lb[:, i], acf_ub[:, i] = confidence_intervals["autocorrelation"]
            (relaxation_timescale_lb[i], relaxation_timescale_ub[i]) = confidence_intervals[
                "relaxation_timescale"
            ]

        # find relaxation time for each trajectory
        for j in range(feats.shape[0]):
            acf_1_crop = autocorrelation_function(feats[j : j + 1], i)
            acf_per_crop[j : j + 1, :, i] = acf_1_crop

            index_positive_crop = lags > 0
            positive_lags_crop = lags[index_positive_crop]
            positive_lags_as_hours_crop = 5 * positive_lags_crop / 60
            acf_positive_lags_crop = acf_1_crop[index_positive_crop]

            _, relaxation_time_crop = fit_exp_decay_and_get_relaxation_timescale(
                acf_positive_lags_crop,
                positive_lags_as_hours_crop,
                exp_decay_func="exponential_decay",
            )
            relaxation_timescale_per_crop[j, i] = relaxation_time_crop

    # cross-correlation
    ccf = np.zeros((num_lags, num_feats))
    ccf_lb = np.zeros((num_lags, num_feats))
    ccf_ub = np.zeros((num_lags, num_feats))

    delta_ccf = np.zeros((num_lags // 2, num_feats))
    delta_ccf_lb = np.zeros((num_lags // 2, num_feats))
    delta_ccf_ub = np.zeros((num_lags // 2, num_feats))

    if max_lag_integrate > num_lags // 2:
        max_lag_integrate = num_lags // 2
        logger.warning(
            "max_lag_integrate is larger than available lags, setting to [ %s ]",
            max_lag_integrate,
        )
    delta_ccf_integral = np.zeros(num_feats)
    delta_ccf_integral_lb = np.zeros(num_feats)
    delta_ccf_integral_ub = np.zeros(num_feats)

    # get the combinations of features for the cross-correlations
    # (we use the indices of the feature labels here because the features
    # themselves are stored in an array)
    feature_indices = range(len(column_names))
    cross_corr_index_combinations = list(combinations(feature_indices, r=2))
    # in `combinations` "r" is the number of elements to include in a combination

    # TODO THIS FOR-LOOP IS SLOW; PARALLELIZE THIS IF POSSIBLE
    for i, (j, k) in enumerate(cross_corr_index_combinations):
        data_feat1 = feats[..., j]
        data_feat2 = feats[..., k]
        ccf[:, i] = cross_correlation_function(data_feat1, data_feat2)
        # get delta CCF = CCF(tau>0) - CCF(tau<0)
        delta_ccf[:, i] = ccf[1 + num_lags // 2 :, i] - ccf[: num_lags // 2, i]
        if bootstrap_samples is not None:
            # calculate bootstrap confidence intervals
            confidence_intervals = bootstrap_cross_correlation_confidence_intervals(
                data_feat1,
                data_feat2,
                max_lag_integrate=max_lag_integrate,
                n_bootstraps=bootstrap_samples,
                max_cores=max_cores,
            )
            ccf_lb[:, i], ccf_ub[:, i] = confidence_intervals["cross_correlation"]
            delta_ccf_lb[:, i], delta_ccf_ub[:, i] = confidence_intervals["delta_cross_correlation"]
            (
                delta_ccf_integral_lb[i],
                delta_ccf_integral_ub[i],
            ) = confidence_intervals["delta_cross_correlation_integral"]

    delta_ccf_integral = cross_correlation_difference_norm(delta_ccf)

    # store results in dict and return that dict
    correlation_dict: dict = {}
    correlation_dict["features"] = column_names
    correlation_dict["lags"] = lags
    correlation_dict["acf"] = acf
    correlation_dict[f"acf_{Column.BootstrapAnalysis.CI_LOWER}"] = acf_lb
    correlation_dict[f"acf_{Column.BootstrapAnalysis.CI_UPPER}"] = acf_ub
    correlation_dict["relaxation_timescales"] = relaxation_timescale
    correlation_dict[f"relaxation_timescales_{Column.BootstrapAnalysis.CI_LOWER}"] = (
        relaxation_timescale_lb
    )
    correlation_dict[f"relaxation_timescales_{Column.BootstrapAnalysis.CI_UPPER}"] = (
        relaxation_timescale_ub
    )
    correlation_dict["ccf"] = ccf
    correlation_dict[f"ccf_{Column.BootstrapAnalysis.CI_LOWER}"] = ccf_lb
    correlation_dict[f"ccf_{Column.BootstrapAnalysis.CI_UPPER}"] = ccf_ub
    correlation_dict["delta_ccf"] = delta_ccf
    correlation_dict[f"delta_ccf_{Column.BootstrapAnalysis.CI_LOWER}"] = delta_ccf_lb
    correlation_dict[f"delta_ccf_{Column.BootstrapAnalysis.CI_UPPER}"] = delta_ccf_ub
    correlation_dict["delta_ccf_integral"] = delta_ccf_integral
    correlation_dict[f"delta_ccf_integral_{Column.BootstrapAnalysis.CI_LOWER}"] = (
        delta_ccf_integral_lb
    )
    correlation_dict[f"delta_ccf_integral_{Column.BootstrapAnalysis.CI_UPPER}"] = (
        delta_ccf_integral_ub
    )
    correlation_dict["max_lag_integrate"] = max_lag_integrate
    correlation_dict["acf_per_crop"] = acf_per_crop
    correlation_dict["relaxation_timescale_per_crop"] = relaxation_timescale_per_crop

    # if the lower confidence interval is higher than the value or the
    # upper confidence interval is lower than the value for any of the
    # metrics (which happens # in DEMO_MODE) then assign that bound to
    # be equal to the value, so that plotting can continue, but log a
    # warning about it
    metrics = ["acf", "ccf", "relaxation_timescales", "delta_ccf", "delta_ccf_integral"]
    for metric in metrics:
        invalid_ci_lower = (
            correlation_dict[metric]
            < correlation_dict[f"{metric}_{Column.BootstrapAnalysis.CI_LOWER}"]
        )
        correlation_dict[f"{metric}_{Column.BootstrapAnalysis.CI_LOWER}"][invalid_ci_lower] = (
            correlation_dict[metric][invalid_ci_lower]
        )
        invalid_ci_upper = (
            correlation_dict[metric]
            > correlation_dict[f"{metric}_{Column.BootstrapAnalysis.CI_UPPER}"]
        )
        correlation_dict[f"{metric}_{Column.BootstrapAnalysis.CI_UPPER}"][invalid_ci_upper] = (
            correlation_dict[metric][invalid_ci_upper]
        )
        logger.warning(
            "Invalid confidence interval bounds found for metric [ %s ] in dataset [ %s ]. "
            "Setting invalid bounds to be equal to the metric value for plotting purposes.",
            metric,
            dataframe[Column.DATASET].iloc[0],
        )
    return correlation_dict


def exponential_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Define exponential decay function for curve fitting."""
    return a * np.exp(-b * x) + c


def double_exponential_decay(
    x: np.ndarray, a1: float, b1: float, a2: float, b2: float, c: float
) -> np.ndarray:
    """Define double exponential decay function for curve fitting."""
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c
