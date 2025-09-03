import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_manifest import (
    df_to_array,
    get_dataframe_for_dynamics_workflows,
    get_pc_column_names,
)
from endo_pipeline.manifests import DataframeManifest

logger = logging.getLogger(__name__)

CROSS_CORR_INDEX_COMBINATIONS = [(0, 1), (0, 2), (1, 2)]


def cross_correlation_function(data_feat1: np.ndarray, data_feat2: np.ndarray) -> float:
    """Get the normalized cross-correlation function (CCF) between two features."""
    num_traj = data_feat1.shape[0]
    num_timepoints = data_feat1.shape[1]

    # pad 0s to nearest power of 2 to 2*num_timepoints-1
    num_pad = 2 ** int(np.ceil(np.log2(2 * num_timepoints - 1)))

    corr_list = []
    for traj_index in range(num_traj):
        data_mean1 = np.mean(data_feat1[traj_index])
        data_stdev1 = np.std(data_feat1[traj_index])
        x_t_i_ctr = data_feat1[traj_index] - data_mean1

        data_mean2 = np.mean(data_feat2[traj_index])
        data_stdev2 = np.std(data_feat2[traj_index])
        x_t_j_ctr = data_feat2[traj_index] - data_mean2

        cf_1 = np.fft.fft(x_t_i_ctr, n=num_pad)
        cf_2 = np.fft.fft(x_t_j_ctr, n=num_pad)
        sf = cf_1.conjugate() * cf_2
        corr = np.fft.ifft(sf).real / (data_stdev1 * data_stdev2 * num_timepoints)
        corr_shifted = np.fft.fftshift(corr)[
            num_pad // 2 - (num_timepoints // 4 - 1) : num_pad // 2 + (num_timepoints // 4)
        ]
        corr_list.append(corr_shifted)

    return sum(corr_list) / num_traj


def autocorrelation_function(data: np.ndarray, component_index: int) -> float:
    """Get the normalized autocorrelation function (ACF) for a specific component."""
    x_t_j = data[..., component_index]
    num_traj = data.shape[0]
    num_timepoints = data.shape[1]

    # pad 0s to nearest power of 2 to 2*num_timepoints-1
    num_pad = 2 ** int(np.ceil(np.log2(2 * num_timepoints - 1)))

    corr_list = []
    for traj_index in range(num_traj):
        data_mean = np.mean(x_t_j[traj_index])
        data_var = np.var(x_t_j[traj_index])
        x_t_j_ctr = x_t_j[traj_index] - data_mean

        cf = np.fft.fft(x_t_j_ctr, n=num_pad)
        sf = cf.conjugate() * cf
        corr = np.fft.ifft(sf).real / (data_var * num_timepoints)
        corr_shifted = np.fft.fftshift(corr)[
            num_pad // 2 - (num_timepoints // 4 - 1) : num_pad // 2 + (num_timepoints // 4)
        ]
        corr_list.append(corr_shifted)

    return sum(corr_list) / num_traj


def bootstrap_cross_correlation_confidence_interval(
    data_feat1: np.ndarray,
    data_feat2: np.ndarray,
    lag: int,
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap the normalized cross-correlation function (CCF) computed from finite data.

    The CCF is computed between between vector components of an ensemble of stationary,
    vector-valued time series data.

    The input two input data arrays are each expected to be of shape (num_samples, num_timepoints).
    That is, the data are assumed to be {num_samples} iid samples, each sampled at
    the same num_timepoints.

    This function resamples the data with replacement to generate bootstrap samples and
    computes the CCF for each bootstrap sample. It then calculates the confidence interval
    for the CCF based on the bootstrap samples.

    Parameters
    ----------
    data_feat1
        Array of shape (num_samples, num_timepoints) containing time series data for the
        first vector component for the CCF.
    data_feat2
        Array of shape (num_samples, num_timepoints) containing time series data for the
        second vector component for the CCF.
    lag
        Time lag (by index) at which to compute the CCF.
    n_bootstraps
        Number of bootstrap samples to generate for the CCF.
    confidence_level
        Confidence interval level (e.g., 95%) to report for the CCF.

    Returns
    -------
    :
        Lower bound of the confidence interval for the CCF.
    :
        Upper bound of the confidence interval for the CCF.
    """

    # Bootstrap the CCF using resampling with replacement
    num_traj = data_feat1.shape[0]
    bootstrap_correlations = []
    for _ in range(n_bootstraps):

        # Random sampling with replacement to generate bootstrap samples
        inds = np.random.choice(num_traj, num_traj, replace=True)
        ds1_resampled = data_feat1[inds]
        ds2_resampled = data_feat2[inds]

        # Calculate cross-correlation for each iteration of resampling and append to list
        bootstrap_correlations.append(cross_correlation_function(ds1_resampled, ds2_resampled))

    # Calculate the lower and upper bounds of the confidence interval
    percentile = (1 - confidence_level) / 2
    lower_bound = np.percentile(bootstrap_correlations, 100 * percentile, axis=0)
    upper_bound = np.percentile(bootstrap_correlations, 100 * (1 - percentile), axis=0)

    return lower_bound, upper_bound


def _compute_correlations_for_one_dataset(
    dataset_name: str,
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    correlation_dict: dict,
    bootstrap_samples: int = 0,
    num_lags_integrate: int = 5,
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
    if num_timepoints % 2 == 0:
        # even number of timepoints
        lags = np.arange(-num_timepoints // 4 + 1, num_timepoints // 4)
    else:
        # odd number of timepoints
        lags = np.arange(-num_timepoints // 4 + 2, num_timepoints // 4)

    num_lags = len(lags)
    # autocorrelation
    acf = np.zeros((num_lags, 3))
    for i in range(3):
        acf[:, i] = autocorrelation_function(feats, i)

    ccf = np.zeros((num_lags, 3))
    ccf_lb = np.zeros((num_lags, 3))
    ccf_ub = np.zeros((num_lags, 3))

    for i, (j, k) in enumerate(CROSS_CORR_INDEX_COMBINATIONS):
        data_feat1 = feats[..., j]
        data_feat2 = feats[..., k]
        ccf[:, i] = cross_correlation_function(data_feat1, data_feat2)
        if bootstrap_samples > 0:
            # calculate bootstrap confidence intervals
            ccf_lb[:, i], ccf_ub[:, i] = bootstrap_cross_correlation_confidence_interval(
                data_feat1,
                data_feat2,
                n_bootstraps=bootstrap_samples,
            )

    # get difference between
    # forward and backward lags
    # leave out zero
    delta_ccf = np.zeros((num_lags // 2, 3))
    for i, _ in enumerate(CROSS_CORR_INDEX_COMBINATIONS):
        delta_ccf[:, i] = abs(ccf[1 + num_lags // 2 :, i] - ccf[: num_lags // 2, i])

    # integrate delta_ccf over first five lags
    if num_lags_integrate > delta_ccf.shape[0]:
        num_lags_integrate = delta_ccf.shape[0]
        logger.warning(
            "num_lags_integrate is larger than available lags, setting to [ %s ]",
            num_lags_integrate,
        )
    delta_ccf_integral = np.trapz(delta_ccf[:num_lags_integrate, :], axis=0)

    # store results in dict of dicts and return updated dict
    correlation_dict["lags"][dataset_name] = lags
    correlation_dict["acf"][dataset_name] = acf
    correlation_dict["ccf"][dataset_name] = ccf
    correlation_dict["ccf_ci_lower"][dataset_name] = ccf_lb
    correlation_dict["ccf_ci_upper"][dataset_name] = ccf_ub
    correlation_dict["delta_ccf"][dataset_name] = delta_ccf
    correlation_dict["delta_ccf_integral"][dataset_name] = delta_ccf_integral
    correlation_dict["num_lags_integrate"][dataset_name] = num_lags_integrate
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
        "ccf": {},
        "ccf_ci_lower": {},
        "ccf_ci_upper": {},
        "delta_ccf": {},
        "delta_ccf_integral": {},
        "num_lags_integrate": {},
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
