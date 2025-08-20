import logging

import numpy as np
from sklearn.decomposition import PCA

from src.endo_pipeline.library.analyze.diffae_manifest import (
    df_to_array,
    get_dataframe_for_dynamics_workflows,
    get_pc_column_names,
)
from src.endo_pipeline.manifests import DataframeManifest

logger = logging.getLogger(__name__)

CROSS_CORR_INDEX_COMBINATIONS = [(0, 1), (0, 2), (1, 2)]


def cross_correlation_function(
    data: np.ndarray, component_1: int, component_2: int, lag: int
) -> float:
    """
    Get the normalized cross-correlation function (CCF) between vector components of an ensemble of
    stationary, vector-valued time series data.

    The input data array is expected to be of shape (num_samples, num_timepoints, num_dim).
    That is, the data are assumed to be {num_samples} iid samples of a num_dim-dimensional
    stationary process, each sampled at the same num_timepoints.

    The cross-correlation function (CCF) is computed for the specified components at the given lag.
    That is, if X(t) is a vector-valued random process, the CCF is defined as the ensemble mean of
    X_{component_1}(t) * X_{component_2}(t + lag). This function computes the CCF using numpy's
    built-in ``corrcoef`` function, which computes the correlation coefficient between two arrays.

    Parameters
    ----------
    data
        Array of shape (num_samples, num_timepoints, num_dim) containing time series data.
    component_1
        Index of the first vector component for the CCF.
    component_2
        Index of the second vector component for the CCF.
    lag
        Time lag (by index) at which to compute the CCF.
    """
    # get number of trajectories
    num_traj = data.shape[0]
    logger.debug("Processing [ %s ] trajectories.", num_traj)

    # check if lag is longer than the time series:
    num_timepoints = data.shape[1]
    if lag >= num_timepoints:
        logger.error(
            "Input lag [ %s ] is longer than the number of time points [ %s ] in the timeseries.",
            lag,
            num_timepoints,
        )
        raise ValueError(
            "Input lag cannot be longer than the number of time points in the timeseries."
        )

    # check if index is greater than dims
    num_dims = data.shape[2]
    if component_1 > num_dims or component_2 > num_dims:
        logger.error(
            "Input component indices [ %s, %s ] are greater than the number of "
            "dimensions [ %s ] in the data.",
            component_1,
            component_2,
            num_dims,
        )
        raise ValueError(
            "Vector component indices cannot be greater than the dimensionality of the data."
        )

    # get slice of data at component 1 (alias x_t_i)
    data_slice_i = data[..., component_1]
    # get slice at component j (alias x_t_j)
    data_slice_j = data[..., component_2]

    # compute mean CCF over all timeseries
    ccf_all = []
    for k in range(num_traj):
        # working with time series k and components i,j
        x_t_i = data_slice_i[k].flatten()
        x_t_j = data_slice_j[k].flatten()

        # stack array of x_t_j from initial point to -lag
        # and array of x_t_i from lag to end point
        if lag > 0:
            stacked_lag = np.array([x_t_j[:-lag], x_t_i[lag:]])
        elif lag < 0:
            stacked_lag = np.array([x_t_j[-lag:], x_t_i[:lag]])
        else:
            stacked_lag = np.array([x_t_j, x_t_i])

        # pass this into np.corrcoeff to get
        # ccf of timeseries x_t with lag time lag
        # across components i and j
        ccf_mat = np.corrcoef(stacked_lag)

        # returns 2x2 matrix, only need the
        # off-diagonal element, which is the
        # correlation between the two components
        # lag and non-lagged
        ccf_ = ccf_mat[0, 1]

        # append to list
        ccf_all.append(ccf_)

    ccf = sum(ccf_all) / len(ccf_all)
    return ccf


def autocorrelation_function(data: np.ndarray, component_index: int, lag: int) -> float:
    """
    Get the normalized autocorrelation function (ACF) for a specific component of an ensemble of
    stationary, vector-valued time series data.

    The input data array is expected to be of shape (num_samples, num_timepoints, num_dim).
    That is, the data are assumed to be {num_samples} iid samples of a num_dim-dimensional
    stationary process, each sampled at the same num_timepoints.

    The autocorrelation function (ACF) is computed for the specified component at the given lag.
    That is, if X(t) is a vector-valued random process, the ACF is defined as the ensemble mean of
    X_{component}(t) * X_{component}(t + lag). This function computes the ACF using numpy's
    built-in ``corrcoef`` function, which computes the correlation coefficient between two arrays.

    Parameters
    ----------
    data
        Array of shape (num_samples, num_timepoints, num_dim) containing time series data.
    component_index
        Index of the vector component for the ACF.
    lag
        Time lag (by index) at which to compute the ACF.
    """

    return cross_correlation_function(data, component_index, component_index, lag)


def _compute_correlations_for_one_dataset(
    dataset_name: str, dataframe_manifest: DataframeManifest, pca: PCA, correlation_dict: dict
) -> dict[str, dict]:
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
        for k in range(num_lags):
            acf[k, i] = autocorrelation_function(feats, i, lags[k])

    ccf = np.zeros((num_lags, 3))
    for i, (j, k) in enumerate(CROSS_CORR_INDEX_COMBINATIONS):
        for lag_index in range(num_lags):
            ccf[lag_index, i] = cross_correlation_function(feats, j, k, lags[lag_index])

    # get difference between
    # forward and backward lags
    # leave out zero
    delta_ccf = np.zeros((num_lags // 2, 3))
    for i, _ in enumerate(CROSS_CORR_INDEX_COMBINATIONS):
        delta_ccf[:, i] = ccf[1 + num_lags // 2 :, i] - ccf[: num_lags // 2, i]

    # store results in dict of dicts and return updated dict
    correlation_dict["lags"][dataset_name] = lags
    correlation_dict["acf"][dataset_name] = acf
    correlation_dict["ccf"][dataset_name] = ccf
    correlation_dict["delta_ccf"][dataset_name] = delta_ccf
    return correlation_dict


def compute_correlation_dict(
    dataset_names: list[str], dataframe_manifest: DataframeManifest, pca: PCA
) -> dict[str, dict]:
    """Compute cross-correlation and autocorrelation for features from each dataset."""
    correlation_dict = {
        "lags": {},
        "acf": {},
        "ccf": {},
        "delta_ccf": {},
    }
    # update dict with correlation functions for each dataset in a loop
    for dataset_name in dataset_names:
        logger.info("Processing dataset [ %s ] for correlation analysis", dataset_name)
        correlation_dict = _compute_correlations_for_one_dataset(
            dataset_name, dataframe_manifest, pca, correlation_dict
        )
    return correlation_dict


def exponential_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Define exponential decay function for curve fitting."""
    return a * np.exp(-b * x)


def power_law_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Define power law decay function for curve fitting."""
    return a * np.power(x, -b)
