from typing import cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_dataframe import get_dataframe_for_dynamics_workflows
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES, NUM_PCS_TO_ANALYZE, ColumnName


def get_bins(
    num_bins: list, data: list[np.ndarray] | None = None, bin_limits: list | None = None
) -> tuple[list, list]:
    """
    Generate histogram bins for computing Kramers-Moyal
    estimates from trajectories, either automatically
    based on data or user-defined bin limits.

    Inputs:
    - Nbins: list of number of bins in each dimension
        (list of length ndim, where ndim is the number
        of dimensions of the feature space)
    - data: list of numpy arrays, each array is the trajectory
        of a single crop in feature space (ndim = len(num_bins))
    - bin_limits: list of tuples, each tuple contains the lower
        and upper bounds for the bins in each dimension

    Either data or bin_limits must be provided.
    If bin_limits provided, data is ignored.

    Outputs:
    - bins: list of numpy arrays, each array contains
        the bin edges for a dimension
    - centers: list of numpy arrays, each array contains
        the center of each bin in a dimension

    If the dimension is 1, bins and centers are still lists (of length 1),
    containing the bin edges and centers for the single dimension.
    """
    if bin_limits is None:  # Automatically determine bins based on data
        if data is None:
            raise ValueError("Please provide data or or upper and lower bounds for bins.")
        ndim = data[0].shape[1]
        assert ndim == len(num_bins), "Number of bins must match number of dimensions in data."
        bins = []
        centers = []
        for i in range(ndim):
            # Get min and max for each dimension across all trajectories
            traj_min = min([traj[:, i].min() for traj in data])
            traj_max = max([traj[:, i].max() for traj in data])
            bin_min, bin_max = traj_min - 0.1, traj_max + 0.1
            my_bins = np.linspace(bin_min, bin_max, num_bins[i] + 1)
            bins.append(my_bins)
            centers.append(0.5 * (my_bins[1:] + my_bins[:-1]))
    else:  # Use user-defined bins
        ndim = len(bin_limits)
        assert ndim == len(num_bins), "Number of bins must match number of dimensions in data."
        bins = []
        centers = []
        for i in range(ndim):
            my_bins = np.linspace(bin_limits[i][0], bin_limits[i][1], num_bins[i] + 1)
            bins.append(my_bins)
            centers.append(0.5 * (my_bins[1:] + my_bins[:-1]))
    return bins, centers


def get_3d_bounds_from_data(
    dataset_names: list[str],
    manifest: DataframeManifest,
    pca: PCA,
    filter_to_valid: bool = True,
    pad: bool = False,
) -> list[np.ndarray]:
    """
    Set bounds for 3D state space based on the bounds
    of the features in the datasets. The 3D state space
    is based on the first three principal components
    of the input pca object, which is fit
    on a fixed set of reference datasets.

    Inputs:
    - dataset_names: list of datasets
    - manifest: manifest of model feature dataframes
    - pca: PCA model to use for transforming the data
    - col_names: which columns to use for bounds
        - "pc": data is coming from a workflow where
            the column names have been re-named to
            reflect that the features are projected
            onto the first three principal components
            (i.e., column names in df pc1, pc2, pc3)
        - "feat": data is coming from a workflow where
            the column names are the original feature names
            and the data have been over-written with the
            features projected onto the full set of
            principal components (i.e., column name feat_i
            indicates projection onto the i-th principal component)
        - this input will become deprecated in the future,
            when the dataframes will always clearly label
            what is an original feature and what is a
            projected feature

    Outputs:
    - bounds: list of numpy arrays with the bounds
        for each dimension in the 3D state space
        - format: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    """
    num_dims = NUM_PCS_TO_ANALYZE  # always 3 for now
    # initialize bounds
    bounds_ = [[np.inf, -np.inf] for _ in range(num_dims)]

    for dataset_name in dataset_names:
        if filter_to_valid:
            filter_dataframe = True
            include_cell_piling = False
            include_not_steady_state = False
        else:
            filter_dataframe = False
            include_cell_piling = True
            include_not_steady_state = True
        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            manifest,
            pca=pca,
            filter_dataframe=filter_dataframe,
            include_cell_piling=include_cell_piling,
            include_not_steady_state=include_not_steady_state,
        )
        # get column names for features
        pc_column_names = DIFFAE_PC_COLUMN_NAMES[:num_dims]
        for j in range(num_dims):
            candidate_min = df[pc_column_names[j]].min()
            candidate_max = df[pc_column_names[j]].max()
            if pad:
                candidate_min = candidate_min - 0.1
                candidate_max = candidate_max + 0.1
            # update bounds for each dimension
            bounds_[j][0] = min(bounds_[j][0], candidate_min)
            bounds_[j][1] = max(bounds_[j][1], candidate_max)

    bounds = [np.array(bounds_[i]) for i in range(num_dims)]

    return bounds


def _get_histogram_by_component_one_dataset(
    df: pd.DataFrame, bin_edges: list[np.ndarray], feat_cols: list[str] | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute histogram of feature data at each timepoint for each latent component.

    Parameters
    ----------
    df
        Feature data for a single dataset.
    bin_edges
        Bin edges for each component.
    feat_cols
        Optional; specific column names of the components to analyze.

    Returns
    -------
    :
        Histogram values for each component as a function of time
    :
        Updated dataframe with bin indices for each crop at each timepoint along each component.
    """
    if feat_cols is None:
        # use all PCA feature columns in the dataframe
        feat_cols_all = DIFFAE_PC_COLUMN_NAMES
        feat_cols = [col for col in feat_cols_all if col in df.columns]

    num_feats = len(feat_cols)
    num_frames = df[ColumnName.TIMEPOINT].nunique()
    num_bins = bin_edges[0].shape[0] - 1  # number of bins is one less than number of edges

    # feats = df_to_array(df_padded, feat_cols)  # get array of just the feature data

    hist_array = np.zeros(
        (num_feats, num_bins, num_frames)
    )  # histogram values for each component as a function of time

    for t, df_frame in df.groupby(ColumnName.TIMEPOINT):
        # loop over latent components
        for dim in range(num_feats):
            feats = df_frame[feat_cols[dim]].to_numpy()
            # compute histogram of feature data along each component
            hist = np.histogram(feats, bins=bin_edges[dim], density=True)[0]
            hist_array[dim, :, t] = hist

            # update the dataframe with column of what bin
            # each crop at frame number t is in
            # along the given latent dimension
            # get the bin index for each crop
            bin_idx = np.digitize(feats, bin_edges[dim]) - 1
            # add the bin index to the dataframe (astype int)
            # restrict to crops at frame number t
            df.loc[df[ColumnName.TIMEPOINT] == t, f"bin_{dim}"] = bin_idx

    # enforce that bin indices are integers
    # this is important for indexing later
    for dim in range(num_feats):
        df[f"bin_{dim}"] = df[f"bin_{dim}"].astype(int)

    # return the histogram array and the updated dataframe
    return hist_array, df


def get_histogram_by_component(
    df: pd.DataFrame,
    num_bins: int,
    bin_limits: list[np.ndarray],
    feat_cols: list[str] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    """
    Get histogram of feature data at each timepoint for each latent component
    across all datasets in the input dataframe.

    Input:
    - df: pd.DataFrame, feature data for multiple datasets
    - num_bins: int, number of bins to use for histogram
        - right now, this is the same for all components
    - bin_limits: list[np.ndarray], bin limits for each component
    - feat_cols: list[str] | None, column names of the features to use
    """
    # get column names for extracting feature data for a single dataset
    if feat_cols is None:
        # use all PCA feature columns in the dataframe
        feat_cols_all = DIFFAE_PC_COLUMN_NAMES
        feat_cols = [col for col in feat_cols_all if col in df.columns]

    num_feats = len(feat_cols)

    # check that bin_limits is provided and matches the number of features
    assert (
        len(bin_limits) == num_feats
    ), f"Number of bin limits ({len(bin_limits)}) must match number of features ({num_feats})"

    # get bin edges for each feature dimension
    bin_edges = [
        get_bins([num_bins], bin_limits=[bin_limits[dim]])[0][0] for dim in range(num_feats)
    ]

    # loop over each dataset in the dataframe
    # get histogram / bin indices for each dataset
    hist_array_list = []
    df_list = []
    for _, df_group in df.groupby(ColumnName.DATASET):
        hist_array, df_group_ = _get_histogram_by_component_one_dataset(
            df_group, bin_edges, feat_cols
        )
        df_list.append(df_group_)
        hist_array_list.append(hist_array)

    df_all_datasets_binned = pd.concat(df_list, ignore_index=True)

    return hist_array_list, bin_edges, df_all_datasets_binned


def _get_index_from_value(val: float, bin_edges_1d: np.ndarray) -> int:
    """
    Given a value and a 1D array of bin edges,
    return the index of the bin that contains that value.

    Example:
    - val = 0.2
    - bin_edges = np.array([0, 0.5, 1])
    - _get_index_from_value(val, bin_edges_1d) = 0
        - i.e., dim 1 = 0.2 falls in the first bin of
         the bin edges for dim 1: [0, 0.5]

    Input:
    - val: float, value to find bin index for
    - dim: int, dimension to find bin index for
    - bin_edges: list[np.ndarray], bin edges for each component
        - this is the same as the output of get_histogram_by_component

    """

    # get the index of the bin that contains the value
    # this is done by finding the index of the first bin edge
    # that is greater than the value
    # and subtracting 1
    bin_idx = cast(int, np.digitize(val, bin_edges_1d) - 1)

    # check if the value is in the last bin
    # if so, set the index to the last bin
    if bin_idx == len(bin_edges_1d) - 1:
        bin_idx = len(bin_edges_1d) - 2

    # check if the value is in the first bin
    # if so, set the index to the first bin
    if bin_idx < 0:
        bin_idx = 0

    # return the index of the bin
    return bin_idx


def get_df_by_bin_value(
    df: pd.DataFrame, pc_axis: int, pc_val: float, bin_edges: list[np.ndarray]
) -> pd.DataFrame:
    """
    Given a dataframe and a latent dimension,
    return the dataframe with only the rows
    such that the value of the component in
    the given latent dimension that falls into
    the bin that contains the given latent value.

    Example:
    - df = pd.DataFrame({'bin_0': [0, 1, 0], 'bin_1': [1, 1, 2]})
    - pc_axis = 0
    - pc_val = 0.2
    - bin_edges = np.array([0, 0.5, 1])
        - _get_index_from_value(latent_val, bin_edges) = 0
        - looking for 'bin_0' == 0
    - get_df_by_bin_value(df, latent_dim, latent_val) =
        pd.DataFrame({'bin_0': [0, 0], 'bin_1': [1, 2]})
        - i.e., the dataframe is filtered to only include rows
        where bin_{latent_dim} is equal to the bin index
        that contains the latent value.

    Input:
    - df: pd.DataFrame, dataframe to filter
    - pc_axis: int, dimension to filter by
    - pc_val: float, value to filter by

    Output:
    - df: pd.DataFrame, filtered dataframe
    """

    # get the bin edges for the given latent dimension
    bin_edges_1d = bin_edges[pc_axis]

    # get the bin index for the given latent value
    # and find the crops that fall into that bin
    bin_idx = _get_index_from_value(pc_val, bin_edges_1d)

    # filter the dataframe to only include rows
    # with bin_{latent_dim} == bin_idx
    df_bin = df.loc[df[f"bin_{pc_axis}"] == bin_idx]

    return df_bin


def get_normalization_constant(p_fit: np.ndarray, dx: list) -> np.ndarray:
    """
    Get normalization constant for stationary probability
    distribution p_fit. The normalization constant is the
    integral of the probability distribution over the state space.

    Inputs:
    - p_fit: np.ndarray, stationary probability
        distribution of the fit SDE model
        - shape N[1] x N[2] x ... x N[ndim]
    - dx: list, bin width in each dimension

    Outputs:
    - c: float, normalization constant
    """
    ndim = len(dx)  # number of dimensions

    # copy p_fit to avoid modifying the original array
    c = p_fit.copy()
    for i in range(ndim):
        # integrate over axis=0 as we marginalize over each dimension
        c = np.trapz(c, dx=dx[i], axis=0)

    return c


def histogramdd(sample: np.ndarray, bins: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    Compute the multidimensional weighted histogram of a sample.

    Allows for a weights matrix to be passed in, which is
    used to weight the samples in each bin.

    This code is a modified version of the histogramdd function
    in Numpy, with the addition of a weights matrix.

    Part of the following code is licensed under the BSD-3 License (from Numpy).

    Inputs:
    - sample: np.ndarray, shape (n, d)
        The input data, where n is the number of samples
        and d is the number of dimensions.
    - bins: list[np.ndarray]
        The bin edges for each dimension. Each element of the list
        is a 1D array of bin edges for that dimension.
    - weights: np.ndarray, shape (n,) or (n, m)
        The weights for each sample.

    Outputs:
    - hist: np.ndarray, shape (nbin,)
        The histogram counts for each bin.
    """

    d = sample.shape[-1]
    # initialize edges, dedges, and nbin
    edges = bins.copy()
    dedges = []
    nbin = np.zeros(d, dtype=int)
    weights = np.asarray(weights)
    for i in range(d):
        nbin[i] = len(edges[i]) + 1
        # check that bins are monotonically increasing
        if np.any(edges[i][:-1] > edges[i][1:]):
            raise ValueError(f"`bins[{i}]` must be monotonically increasing, when an array")
        # increase bin count by 1 to include outliers
        nbin[i] = len(edges[i]) + 1
        # get the width of each bin
        dedges.append(np.diff(edges[i]))

    m = len(bins)
    if m != d:
        raise ValueError("The dimension of bins must be equal to the dimension of the " " sample x")

    # Get the histogram counts.
    hist: np.ndarray = _get_bin_counts(sample, weights, edges, d, nbin)

    # Reshape the histogram matrix to the correct shape.
    if weights.ndim == 1:
        hist = hist.reshape(nbin)
    else:
        hist = hist.reshape((weights.shape[0], *nbin))

    # Remove outliers (indices 0 and -1 for each dimension).
    core: tuple[slice, ...] = d * (slice(1, -1),)

    # slice the histogram to remove outliers
    # Tell MyPy to ignore the type error here,
    # doesn't like indexing via ellipsis
    hist = hist[..., *core]  # type: ignore

    return hist


def _bincount(x: np.ndarray, weights: np.ndarray, minlength: int = 0) -> np.ndarray:
    """Get the weighted counts of the input array x."""
    return np.array([np.bincount(x, w, minlength=minlength) for w in weights])


def _get_bin_counts(
    sample: np.ndarray,
    weights: np.ndarray,
    edges: list[np.ndarray],
    d: int,
    nbin: np.ndarray,
) -> np.ndarray:
    """Get weighted bin counts for the input sample."""
    # Compute the bin number each sample falls into.
    n_count = tuple(np.searchsorted(edges[i], sample[:, i], side="right") for i in range(d))

    # Using searchsorted, values that fall on an
    # edge are put in the right bin.
    # For the rightmost bin, we want values equal
    # to the right edge to be counted in the last bin,
    # and not as an outlier.
    for i in range(d):
        # Find which points are on the rightmost edge.
        on_edge = sample[:, i] == edges[i][-1]
        # Shift these points one bin to the left.
        n_count[i][on_edge] -= 1

    # These next two lines assign the
    # correct bin count to the histogram.

    # Compute the sample indices in the flattened histogram matrix.
    # Ensure n_count is a tuple of integer arrays
    n_count = tuple(arr.astype(int) for arr in n_count)
    xy = np.ravel_multi_index(n_count, tuple(map(int, nbin)))

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = _bincount(xy, weights, minlength=int(np.prod(nbin)))
    return hist
