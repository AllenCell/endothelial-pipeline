import logging
from typing import Any, Literal

import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_manifest import (
    df_to_array,
    get_dataframe_for_dynamics_workflows,
    get_pc_column_names,
)
from endo_pipeline.manifests import DataframeManifest

logger = logging.getLogger(__name__)

CROSS_CORR_INDEX_COMBINATIONS = [(0, 1), (0, 2), (1, 2)]
# use lags going from - to + {num_timepoints}//NUM_TIMEPOINT_FRAC for CCF/ACF calculation
NUM_TIMEPOINT_FRAC = 3

# set upper bound of integration for delta CCF integral calculation
MAX_LAG_INTEGRATE = 5


def cross_correlation_function(data_feat1: np.ndarray, data_feat2: np.ndarray) -> np.ndarray:
    """Get the normalized cross-correlation function (CCF) between two features."""
    num_traj = data_feat1.shape[0]
    num_timepoints = data_feat1.shape[1]

    # Get nearest power of 2 greater than 2*num_timepoints-1.
    # This is to pass into np.fft.fft to pad the signal with
    # zeros for efficient FFT computation (Cooley-Tukey algorithm)
    num_pad = 2 ** int(np.ceil(np.log2(2 * num_timepoints - 1)))

    for traj_index in range(num_traj):
        # Center data by subtracting mean, get standard deviation
        # for normalization of CCF.
        data_mean1 = np.mean(data_feat1[traj_index])
        data_stdev1 = np.std(data_feat1[traj_index])
        x_t_i_ctr = data_feat1[traj_index] - data_mean1

        data_mean2 = np.mean(data_feat2[traj_index])
        data_stdev2 = np.std(data_feat2[traj_index])
        x_t_j_ctr = data_feat2[traj_index] - data_mean2

        # By the convolution theorem, the CCF is the inverse FFT of the cross power spectrum
        # (i.e., X1^{*}(f) * X2(f) where X1 and X2 are the FFTs of the two signals).

        # Get the FFT of the centered data, padding with zeros to length num_pad.
        cf_1 = np.fft.fft(x_t_i_ctr, n=num_pad)
        cf_2 = np.fft.fft(x_t_j_ctr, n=num_pad)
        # Compute the cross power spectrum of the padded signals (normalized by num_timepoints).
        sf = cf_1.conjugate() * cf_2 / num_timepoints

        # Compute the inverse FFT of the power spectrum to get the CCF,
        # normalizing by product of standard deviations (definition of scaled CCF)
        corr_unshifted = np.fft.ifft(sf).real / (data_stdev1 * data_stdev2)

        # Shift the CCF so that zero lag is in the center of the array.
        corr_shifted = np.fft.fftshift(corr_unshifted)
        # Extract the middle half of the CCF (corresponding to lags going from
        # - to + num_timepoints//NUM_TIMEPOINT_FRAC) to get the actual CCF of the unpadded signal.
        max_lag = num_timepoints // NUM_TIMEPOINT_FRAC
        index_lb = num_pad // 2 - max_lag
        index_ub = num_pad // 2 + max_lag + 1
        corr = corr_shifted[index_lb:index_ub]

        # Running sum over trajectories to get average.
        if traj_index == 0:
            corr_sum = corr
        else:
            corr_sum = corr_sum + corr

    # Return average over number of trajectories.
    return corr_sum / num_traj


def autocorrelation_function(data: np.ndarray, component_index: int) -> np.ndarray:
    """Get the normalized autocorrelation function (ACF) for a specific component."""
    # Extract the specified component from the data array.
    x_t_j = data[..., component_index]

    # Pass to cross_correlation_function with itself to get ACF.
    return cross_correlation_function(x_t_j, x_t_j)


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

    if exp_decay_func == "exponential_decay":
        p0 = [0.5, 0.5, 0.5]  # initial guess for a, b, and c
        exp_fit, _ = curve_fit(exponential_decay, lags, acf, maxfev=maxfev, p0=p0)
        relaxation_time = 1 / exp_fit[1]
    else:
        p0 = [0.5, 0.5, 0.5, 0.5, 0.5]  # initial guess for a1, b1, a2, b2, and c
        exp_fit, _ = curve_fit(double_exponential_decay, lags, acf, maxfev=maxfev, p0=p0)
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


def bootstrap_autocorrelation_confidence_intervals(
    data: np.ndarray,
    component_index: int,
    lags: np.ndarray,
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
) -> dict[str, tuple]:
    """
    Bootstrap the normalized autocorrelation function (ACF) computed from finite data.

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

    Returns
    -------
    :
        Dictionary containing lower and upper bounds of the confidence intervals for:
        - autocorrelation: ACF(tau)
        - relaxation_timescale: Decay coefficient from exponential fit to ACF.
    """

    # Bootstrap the CCF using resampling with replacement
    num_traj = data.shape[0]
    bootstrap_autocorrelations = []
    bootstrap_relaxation_timescales = []
    for _ in range(n_bootstraps):
        # Random sampling with replacement to generate bootstrap samples
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

        # store results
        bootstrap_autocorrelations.append(acf)
        bootstrap_relaxation_timescales.append(relaxation_time)

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


def bootstrap_cross_correlation_confidence_intervals(
    data_feat1: np.ndarray,
    data_feat2: np.ndarray,
    n_bootstraps: int = 200,
    max_lag_integrate: int = MAX_LAG_INTEGRATE,
    confidence_level: float = 0.95,
) -> dict[str, tuple]:
    """
    Bootstrap the normalized cross-correlation function (CCF) computed from finite data.

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
    num_traj = data_feat1.shape[0]
    bootstrap_correlations = []
    bootstrap_correlation_diffs = []
    bootstrap_correlation_diff_integrals = []
    for _ in range(n_bootstraps):
        # Random sampling with replacement to generate bootstrap samples
        inds = np.random.choice(num_traj, num_traj, replace=True)
        ds1_resampled = data_feat1[inds]
        ds2_resampled = data_feat2[inds]

        # Calculate cross-correlation for each iteration of resampling and append to list
        ccf = cross_correlation_function(ds1_resampled, ds2_resampled)
        num_lags = len(ccf)
        delta_ccf = ccf[1 + num_lags // 2 :] - ccf[: num_lags // 2]
        delta_ccf_integral = cross_correlation_difference_norm(delta_ccf, max_lag_integrate)
        bootstrap_correlations.append(ccf)
        bootstrap_correlation_diffs.append(delta_ccf)
        bootstrap_correlation_diff_integrals.append(delta_ccf_integral)

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


def _compute_correlations_for_one_dataset(
    dataset_name: str,
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    correlation_dict: dict,
    bootstrap_samples: int = 0,
    max_lag_integrate: int = MAX_LAG_INTEGRATE,
) -> dict[str, dict[str, Any]]:
    """Compute cross-correlation and autocorrelation for features from one dataset."""

    # try to get dataframe for the given dataset
    # if it does not exist, skip this dataset, return dict as is
    try:
        df = get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest, pca)
    except KeyError:
        logger.warning(
            "Dataset [ %s ] not found in the manifest, skipping for this workflow.", dataset_name
        )
        return correlation_dict

    feat_cols = get_pc_column_names(df, pc_axes=[0, 1, 2])

    # get feature data
    feats = df_to_array(df, feat_cols)

    num_timepoints = feats.shape[1]
    # make sure lags are symmetric around zero
    max_lags = num_timepoints // NUM_TIMEPOINT_FRAC
    lags = np.arange(-max_lags, max_lags + 1)

    num_lags = len(lags)
    # autocorrelation
    acf = np.zeros((num_lags, 3))
    acf_lb = np.zeros((num_lags, 3))
    acf_ub = np.zeros((num_lags, 3))
    relaxation_timescale_lb = np.zeros(3)
    relaxation_timescale_ub = np.zeros(3)
    for i in range(3):
        acf[:, i] = autocorrelation_function(feats, i)
        if bootstrap_samples > 0:
            # calculate bootstrap confidence intervals for ACF and relaxation timescale
            confidence_intervals = bootstrap_autocorrelation_confidence_intervals(
                feats, i, lags, n_bootstraps=bootstrap_samples
            )
            acf_lb[:, i], acf_ub[:, i] = confidence_intervals["autocorrelation"]
            (relaxation_timescale_lb[i], relaxation_timescale_ub[i]) = confidence_intervals[
                "relaxation_timescale"
            ]

    # cross-correlation
    ccf = np.zeros((num_lags, 3))
    ccf_lb = np.zeros((num_lags, 3))
    ccf_ub = np.zeros((num_lags, 3))

    delta_ccf = np.zeros((num_lags // 2, 3))
    delta_ccf_lb = np.zeros((num_lags // 2, 3))
    delta_ccf_ub = np.zeros((num_lags // 2, 3))

    if max_lag_integrate > num_lags // 2:
        max_lag_integrate = num_lags // 2
        logger.warning(
            "max_lag_integrate is larger than available lags, setting to [ %s ]",
            max_lag_integrate,
        )
    delta_ccf_integral = np.zeros(3)
    delta_ccf_integral_lb = np.zeros(3)
    delta_ccf_integral_ub = np.zeros(3)

    for i, (j, k) in enumerate(CROSS_CORR_INDEX_COMBINATIONS):
        data_feat1 = feats[..., j]
        data_feat2 = feats[..., k]
        ccf[:, i] = cross_correlation_function(data_feat1, data_feat2)
        # get delta CCF = CCF(tau>0) - CCF(tau<0)
        delta_ccf[:, i] = ccf[1 + num_lags // 2 :, i] - ccf[: num_lags // 2, i]
        if bootstrap_samples > 0:
            # calculate bootstrap confidence intervals
            confidence_intervals = bootstrap_cross_correlation_confidence_intervals(
                data_feat1,
                data_feat2,
                max_lag_integrate=max_lag_integrate,
                n_bootstraps=bootstrap_samples,
            )
            ccf_lb[:, i], ccf_ub[:, i] = confidence_intervals["cross_correlation"]
            delta_ccf_lb[:, i], delta_ccf_ub[:, i] = confidence_intervals["delta_cross_correlation"]
            (
                delta_ccf_integral_lb[i],
                delta_ccf_integral_ub[i],
            ) = confidence_intervals["delta_cross_correlation_integral"]

    delta_ccf_integral = cross_correlation_difference_norm(delta_ccf)

    # store results in dict of dicts and return updated dict
    correlation_dict["lags"][dataset_name] = lags
    correlation_dict["acf"][dataset_name] = acf
    correlation_dict["acf_ci_lower"][dataset_name] = acf_lb
    correlation_dict["acf_ci_upper"][dataset_name] = acf_ub
    correlation_dict["relaxation_timescales_ci_lower"][dataset_name] = relaxation_timescale_lb
    correlation_dict["relaxation_timescales_ci_upper"][dataset_name] = relaxation_timescale_ub
    correlation_dict["ccf"][dataset_name] = ccf
    correlation_dict["ccf_ci_lower"][dataset_name] = ccf_lb
    correlation_dict["ccf_ci_upper"][dataset_name] = ccf_ub
    correlation_dict["delta_ccf"][dataset_name] = delta_ccf
    correlation_dict["delta_ccf_ci_lower"][dataset_name] = delta_ccf_lb
    correlation_dict["delta_ccf_ci_upper"][dataset_name] = delta_ccf_ub
    correlation_dict["delta_ccf_integral"][dataset_name] = delta_ccf_integral
    correlation_dict["delta_ccf_integral_ci_lower"][dataset_name] = delta_ccf_integral_lb
    correlation_dict["delta_ccf_integral_ci_upper"][dataset_name] = delta_ccf_integral_ub
    correlation_dict["max_lag_integrate"][dataset_name] = max_lag_integrate
    return correlation_dict


def compute_correlation_dict(
    dataset_names: list[str],
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    bootstrap_samples: int = 0,
) -> dict[str, dict]:
    """Compute cross-correlation and autocorrelation for features from each dataset."""
    correlation_dict: dict[str, dict[str, np.ndarray]] = {
        "lags": {},
        "acf": {},
        "acf_ci_lower": {},
        "acf_ci_upper": {},
        "relaxation_timescales_ci_lower": {},
        "relaxation_timescales_ci_upper": {},
        "ccf": {},
        "ccf_ci_lower": {},
        "ccf_ci_upper": {},
        "delta_ccf": {},
        "delta_ccf_ci_lower": {},
        "delta_ccf_ci_upper": {},
        "delta_ccf_integral": {},
        "delta_ccf_integral_ci_lower": {},
        "delta_ccf_integral_ci_upper": {},
        "max_lag_integrate": {},
        "relaxation_timescales": {},
    }
    # update dict with correlation functions for each dataset in a loop
    for dataset_name in dataset_names:
        correlation_dict = _compute_correlations_for_one_dataset(
            dataset_name, dataframe_manifest, pca, correlation_dict, bootstrap_samples
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
