"""Methods for computing autocorrelation and cross-correlation functions from time series data."""

import logging
from collections.abc import Callable, Sequence
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.numerics.forward_difference import (
    compute_forward_differences_along_trajectory,
)
from endo_pipeline.settings.autocorrelations import NUM_TIMEPOINT_FRAC
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import POLAR_ANGLE_PERIOD, TIME_STEP_IN_HOURS

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

    # Ensure x_t_j is 2D (num_samples, num_timepoints). If a single trajectory
    # is passed with shape (num_timepoints,), reshape to (1, num_timepoints).
    if x_t_j.ndim == 1:
        x_t_j = x_t_j[np.newaxis, :]

    # Pass to cross_correlation_function with itself to get ACF.
    return cross_correlation_function(x_t_j, x_t_j, lag_cutoff_fraction=lag_cutoff_fraction)


def fit_exp_decay(
    acf: np.ndarray,
    lags: np.ndarray,
    maxfev: int = 10000,
    p0: Sequence[float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Fit exponential decay to ACF and return fit parameters and relaxation timescale."""

    # get indices where both lags and acf are finite, as required for input to curve_fit function
    valid_indices = np.isfinite(acf) & np.isfinite(lags)

    acf_valid = acf[valid_indices]
    lags_valid = lags[valid_indices]

    exp_fit, _ = curve_fit(exponential_decay, lags_valid, acf_valid, maxfev=maxfev, p0=p0)

    return exp_fit


def _fill_missing_timepoints_with_nans(
    data_crop: pd.DataFrame, all_timepoints: np.ndarray
) -> pd.DataFrame:
    """Fill missing timepoints in a crop dataframe with NaN values."""
    if data_crop[Column.CROP_INDEX].nunique() != 1:
        raise ValueError("Dataframe contains multiple crop indices.")

    # sort by timepoint to ensure correct order before reindexing
    data_crop = data_crop.sort_values(by=Column.TIMEPOINT)

    # preserve the crop index value so it survives the reindex step
    crop_index_value = data_crop[Column.CROP_INDEX].iloc[0]

    # reindex dataframe to include all timepoints in full range
    data_crop_filled = data_crop.set_index(Column.TIMEPOINT).reindex(all_timepoints)

    # restore timepoint column and fill CROP_INDEX for NaN-inserted rows
    data_crop_filled = data_crop_filled.reset_index()
    data_crop_filled[Column.CROP_INDEX] = crop_index_value

    return data_crop_filled


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
    metadata_dict
        Optional dictionary of additional metadata to add as columns to the output
        dataframe (e.g. dataset name, shear stress).

    """
    # check that required columns are present in the dataframe
    required_columns = [*column_names, Column.CROP_INDEX, Column.DATASET]
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
        data_crop_filled = _fill_missing_timepoints_with_nans(data_crop, all_timepoints)
        data_filled_list.append(data_crop_filled)

    dataframe_filled = pd.concat(data_filled_list, ignore_index=True)

    # use default lag cutoff fraction to determine lags for ACF calculation,
    # which determines the number of lags to include in the output dataframe
    num_timepoints = len(all_timepoints)
    max_lags = num_timepoints // NUM_TIMEPOINT_FRAC
    lags = np.arange(-max_lags, max_lags + 1)

    # dataframe as array of shape (num_crops, num_timepoints, num_feats) for the
    # current feature, with missing timepoints filled with NaNs
    acf_dataframe_list = []
    for i, column_name in enumerate(column_names):
        acf_per_crop = []
        for _, df_crop in dataframe_filled.groupby(Column.CROP_INDEX):
            feats = df_crop[column_name].to_numpy()[np.newaxis, :, np.newaxis]
            acf_per_crop.append(
                autocorrelation_function(feats, 0, lag_cutoff_fraction=NUM_TIMEPOINT_FRAC)
            )

        # take mean and percentiles across crops for each lag to get mean and
        # confidence intervals for the ACF at each lag across the population of
        # single crop trajectories
        acf_mean_all_lags = np.nanmean(acf_per_crop, axis=0)
        acf_lower_bound_all_lags = np.nanpercentile(acf_per_crop, lower_percentile, axis=0)
        acf_upper_bound_all_lags = np.nanpercentile(acf_per_crop, upper_percentile, axis=0)

        # only keep positive lags for the output dataframe since the ACF is
        # symmetric around zero and we are primarily interested in the decay of
        # the ACF at positive lags
        positive_lags = lags[lags > 0]
        acf_mean = acf_mean_all_lags[lags > 0]
        acf_lower_bound = acf_lower_bound_all_lags[lags > 0]
        acf_upper_bound = acf_upper_bound_all_lags[lags > 0]

        # fit exponential decay to the mean ACF at positive lags and get the
        # evaluated exponential fit curve at the positive lags to add to the
        # output dataframe
        exp_fit = fit_exp_decay(acf_mean, positive_lags)
        exp_fit_evaluated = exponential_decay(positive_lags, *exp_fit)

        acf_dataframe_list.append(
            pd.DataFrame(
                {
                    Column.AutoCorrelation.FEATURE: column_names[i],
                    Column.AutoCorrelation.LAG: positive_lags,
                    Column.AutoCorrelation.ACF_MEAN: acf_mean,
                    Column.AutoCorrelation.ACF_LOWER_PERCENTILE: acf_lower_bound,
                    Column.AutoCorrelation.ACF_UPPER_PERCENTILE: acf_upper_bound,
                    Column.AutoCorrelation.EXPONENTIAL_FIT: exp_fit_evaluated,
                }
            )
        )

    acf_dataframe = pd.concat(acf_dataframe_list, ignore_index=True)

    if metadata_dict is not None:
        for key in metadata_dict:
            acf_dataframe[key] = metadata_dict[key]

    return acf_dataframe


def exponential_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Define exponential decay function for curve fitting."""
    return a * np.exp(-b * x) + c


def get_trajectories_and_differences_for_noise_correlations(
    df_flow: pd.DataFrame, column_names: list[str], max_num_patches: int | None = None
):
    """
    Build list of trajectories and their forward differences for noise correlation analysis.

    Parameters
    ----------
    df_flow
        Dataframe of features for the current flow condition.
    column_names
        List of feature column names to get trajectories for.
    max_num_patches
        Maximum number of patches to process. If None, process all patches.
    """
    # Loop over each patch in the dataset and compute the forward differences
    num_patches_processed = 0
    traj_list = []
    d_traj_list = []
    for patch_idx, df_patch_ in df_flow.groupby(Column.CROP_INDEX):
        if max_num_patches is not None and num_patches_processed >= max_num_patches:
            logger.debug("Reached maximum number of patches: %d", max_num_patches)
            break

        # sort by timepoint to ensure that trajectory is in correct order before
        # computing differences
        df_patch = df_patch_.sort_values(by=Column.TIMEPOINT)

        # compute forward differences along trajectory for this crop, and filter
        # to keep only differences between timepoints that are separated by
        # time_lag number of frames (accounts for any missing timepoints in the
        # trajectory, for example due to outlier filtering)
        filtered_traj, filtered_d_traj = compute_forward_differences_along_trajectory(
            df_patch, column_names
        )

        # if the returned difference array is empty, skip this trajectory
        if filtered_traj.empty or filtered_d_traj.empty:
            logger.warning(
                "Skipping patch with empty trajectory or difference arrays: [ %s ]",
                patch_idx,
            )
            continue

        traj_list.append(filtered_traj)
        d_traj_list.append(filtered_d_traj)
        num_patches_processed += 1

    return traj_list, d_traj_list


class NoiseCorrelationCentering(StrEnum):
    """
    Class representing the different centering methods for computing noise
    correlations.
    """

    MEAN_SUBTRACTED = "mean-subtracted"
    """Center differences by subtracting the ensemble mean of the differences at
    the given timepoint."""

    DRIFT_SUBTRACTED = "drift-subtracted"
    """Center differences by subtracting the estimated drift component."""


def compute_centered_residuals(
    traj_list: list[pd.DataFrame],
    d_traj_list: list[pd.DataFrame],
    column_names: list[str],
    timepoints_array: np.ndarray,
    vector_field: Callable,
    centering_method: NoiseCorrelationCentering,
) -> np.ndarray:
    """
    Build the array of drift-removed residuals eta(t) for every patch.

    The residual is the part of the observed forward difference that is not
    explained by the deterministic drift, so under the modeling assumption of an
    Euler-Maruyama discretized SDE it is the realized noise increment. The drift
    is removed either by subtracting the ensemble mean displacement at each
    timepoint or by subtracting the drift predicted by the estimated vector
    field.

    Parameters
    ----------
    traj_list
        List of dataframes containing trajectories for each patch.
    d_traj_list
        List of dataframes containing forward differences for each patch.
    column_names
        List of feature column names.
    timepoints_array
        Array of timepoints to consider.
    vector_field
        Callable vector field used to estimate the drift component.
    centering_method
        Method used to center the differences.

    Returns
    -------
    :
        Array of shape (num_patches, num_timepoints, num_features) containing
        the residuals, with NaN at timepoints where a patch has no difference.
    """
    n_dim = len(column_names)
    n_timepoints = timepoints_array.shape[0]
    n_patches = len(traj_list)
    diff_column_names = [f"{col}{Column.DiffAEData.DIFFERENCE_SUFFIX}" for col in column_names]

    mean_dx_t = None
    if centering_method == NoiseCorrelationCentering.MEAN_SUBTRACTED:
        # Ensemble mean of the forward differences across patches at each timepoint
        mean_dx_t = (
            pd.concat(d_traj_list)
            .groupby(Column.TIMEPOINT)[diff_column_names]
            .mean()
            .reindex(timepoints_array)
            .to_numpy()
        )

    residuals = np.full((n_patches, n_timepoints, n_dim), np.nan)
    for patch_index, (filtered_traj, filtered_d_traj) in enumerate(
        zip(traj_list, d_traj_list, strict=True)
    ):
        # Extract the timepoints, values, and differences for this patch
        patch_timepoints_x_t = filtered_traj[Column.TIMEPOINT].to_numpy()
        patch_timepoints_dx_t = filtered_d_traj[Column.TIMEPOINT].to_numpy()
        patch_x_t = filtered_traj[column_names].to_numpy()
        patch_dx_t = filtered_d_traj[diff_column_names].to_numpy()

        # place this patch's trajectory values and differences on the
        # shared timepoint axis, leaving NaN at timepoints where the
        # patch has no difference/value
        x_t = np.full((n_timepoints, n_dim), np.nan)
        x_t[np.searchsorted(timepoints_array, patch_timepoints_x_t)] = patch_x_t
        dx_t = np.full((n_timepoints, n_dim), np.nan)
        dx_t[np.searchsorted(timepoints_array, patch_timepoints_dx_t)] = patch_dx_t

        if mean_dx_t is not None:
            residuals[patch_index] = dx_t - mean_dx_t
        else:
            residuals[patch_index] = dx_t - vector_field(x_t) * TIME_STEP_IN_HOURS

    return residuals


def compute_two_timepoint_noise_correlations(
    traj_list: list[pd.DataFrame],
    d_traj_list: list[pd.DataFrame],
    column_names: list[str],
    timepoints_array: np.ndarray,
    vector_field: Callable,
    centering_method: NoiseCorrelationCentering,
) -> np.ndarray | None:
    """
    Compute two-timepoint cross correlations of the noise present in
    observations of feature trajectories and their forward differences.

    Parameters
    ----------
    traj_list
        List of dataframes containing trajectories for each patch.
    d_traj_list
        List of dataframes containing forward differences for each patch.
    column_names
        List of feature column names.
    timepoints_array
        Array of timepoints to consider for the correlations.
    vector_field
        Callable vector field used to estimate the drift component.
    centering_method
        Method used to center the differences when computing noise correlations.

    Returns
    -------
    :
        Array containing the two-timepoint noise correlations.
    """

    if centering_method not in NoiseCorrelationCentering:
        logger.error(
            "Invalid centering method: %s. Must be one of %s.",
            centering_method,
            list(NoiseCorrelationCentering),
        )
        return

    # Build arrays to store results
    n_dim = len(column_names)
    n_timepoints = timepoints_array.shape[0]
    cross_correlations = np.zeros((n_timepoints, n_timepoints, n_dim, n_dim))
    n_points = np.zeros((n_timepoints, n_timepoints))

    residuals = compute_centered_residuals(
        traj_list=traj_list,
        d_traj_list=d_traj_list,
        column_names=column_names,
        timepoints_array=timepoints_array,
        vector_field=vector_field,
        centering_method=centering_method,
    )

    for patch_residuals in residuals:
        # mask to valid timepoints
        is_valid = ~np.isnan(patch_residuals).any(axis=1)
        eta_t = np.where(is_valid[:, np.newaxis], patch_residuals, 0.0)

        # Compute two timepoint cross correlation: R_ij(t, t') = < eta_i(t) eta_j(t') >
        # Use Einstein summation to efficiently compute this numerically.
        cross_correlations += np.einsum("ti,sj->tsij", eta_t, eta_t)
        n_points += np.outer(is_valid, is_valid)

    # Normalize accumulated sums by the number of contributing patches
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_correlations /= n_points[:, :, np.newaxis, np.newaxis]

    return cross_correlations


def compute_lagged_covariances(
    traj_list: list[pd.DataFrame],
    d_traj_list: list[pd.DataFrame],
    column_names: list[str],
    timepoints_array: np.ndarray,
    vector_field: Callable,
    centering_method: NoiseCorrelationCentering,
    max_lag: int = 10,
) -> np.ndarray:
    """
    Compute pooled lagged covariance matrices of the residuals.

    The covariance at lag ``tau`` is ``R_ij(tau) = < eta_i(t) eta_j(t + tau) >``,
    pooled over every patch and every timepoint at which both members of the
    pair are observed. Pairs that straddle a missing timepoint are excluded
    rather than zero filled, so that gaps in a track do not bias the estimate
    toward zero.

    Parameters
    ----------
    traj_list
        List of dataframes containing the trajectories for each patch.
    d_traj_list
        List of dataframes containing the forward differences of the trajectories.
    column_names
        List of column names corresponding to the features in the trajectories.
    timepoints_array
        Array of timepoints corresponding to the rows in the trajectory dataframes.
    vector_field
        Callable representing the vector field used for centering the residuals.
    centering_method
        Method used to center the residuals.
    max_lag
        Largest lag, in frames, to evaluate.

    Returns
    -------
    :
        Covariance matrices of shape (max_lag + 1, num_features, num_features).
    """

    if centering_method not in NoiseCorrelationCentering:
        logger.error(
            "Invalid centering method: %s. Must be one of %s.",
            centering_method,
            list(NoiseCorrelationCentering),
        )
        return

    # Get centered feature displacement residuals
    n_dim = len(column_names)
    n_timepoints = timepoints_array.shape[0]
    max_lag = int(min(max_lag, n_timepoints - 1))

    residuals = compute_centered_residuals(
        traj_list=traj_list,
        d_traj_list=d_traj_list,
        column_names=column_names,
        timepoints_array=timepoints_array,
        vector_field=vector_field,
        centering_method=centering_method,
    )

    is_valid = ~np.isnan(residuals).any(axis=2)
    centered = residuals - np.nanmean(residuals, axis=(0, 1))
    # zero fill so that excluded entries contribute nothing to the sums below
    centered = np.where(is_valid[..., np.newaxis], centered, 0.0)

    covariances = np.full((max_lag + 1, n_dim, n_dim), np.nan)
    for lag in range(max_lag + 1):
        leading = centered[:, : n_timepoints - lag, :]
        trailing = centered[:, lag:, :]
        n_pairs = int((is_valid[:, : n_timepoints - lag] & is_valid[:, lag:]).sum())
        if n_pairs == 0:
            continue
        covariances[lag] = np.einsum("pti,ptj->ij", leading, trailing) / n_pairs

    return covariances


def normalize_lagged_covariances(covariances: np.ndarray) -> np.ndarray:
    """Rescale lagged covariances to correlations in [-1, 1]."""
    sigma = np.sqrt(np.diag(covariances[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariances / np.outer(sigma, sigma)


def compute_measurement_noise_variance(covariances: np.ndarray) -> np.ndarray:
    """
    Estimate the variance of the independent error on each observed coordinate.

    Independent error enters the forward difference as ``eta(t) = s(t) + e(t +
    1) - e(t)``, where s(t) is the dynamical noise (true signal) and e(t) is the
    independent measurement error. The error variance is therefore equal to the
    the lag-one covariance ``cov(eta(t), eta(t + 1))``. The estimate only has
    meaning when that covariance is negative, and is reported as NaN otherwise.

    Parameters
    ----------
    covariances
        Lagged covariance matrices of shape (max_lag + 1, num_features,
        num_features).

    Returns
    -------
    :
        Estimated measurement noise variance for each feature, of shape
        (num_features,).
    """
    if covariances.shape[0] < 2:
        raise ValueError(
            "Covariances must have at least two lags to estimate lag-one measurement noise variance."
        )
    lag_one_covariance = np.einsum("ii->i", covariances[1])
    return np.where(lag_one_covariance < 0, lag_one_covariance, np.nan)
