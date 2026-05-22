"""Methods for computing autocorrelation and cross-correlation functions from time series data."""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.settings.autocorrelations import NUM_TIMEPOINT_FRAC
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import POLAR_ANGLE_PERIOD

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
