import logging
from typing import cast

import numpy as np
import pandas as pd

from endo_pipeline.library.analyze.polar_coords import rewrap_polar_angle
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.flow_field_3d import PAD_BINS_FLOAT

logger = logging.getLogger(__name__)


def circpercentile(
    angles: np.ndarray, q: float, polar_range: tuple[float, float] = (0, np.pi)
) -> float:
    """
    Compute the q-th percentile of circular data.

    Parameters
    ----------
    angles
        1D array of circular data (e.g., angles in radians).
    q
        Percentile to compute (between 0 and 100).
    polar_range
        Tuple specifying the circular range of the data (e.g., (0,
        np.pi) for angles in radians).
    """

    sorted_angles = np.sort(angles)

    # Find largest gap (including wrap-around gap)
    period = polar_range[1] - polar_range[0]
    angle_diffs = np.diff(sorted_angles, append=sorted_angles[0] + period)
    where_largest_diff = np.argmax(angle_diffs)

    # Cut at end of largest gap; shift so data are contiguous on line
    angle_cut = (sorted_angles[where_largest_diff] + angle_diffs[where_largest_diff]) % period
    contiguous_angles = np.mod(angles - angle_cut, period)

    # Ordinary percentile in linear space
    angle_percentile = np.percentile(contiguous_angles, q)

    # Shift back to circular space, and rewrap to original polar range
    return rewrap_polar_angle(angle_percentile + angle_cut, polar_range)


def get_bins(
    bin_widths: tuple[float, ...],
    data: np.ndarray | None = None,
    bin_limits: list[tuple[float, float]] | None = None,
    pad: float = PAD_BINS_FLOAT,
    lower_percentile: float | None = None,
    upper_percentile: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate histogram bins either automatically based on data or user-defined
    bin limits.

    **Binning Options:**

    If `bin_limits` is not provided, the function automatically determines bin
    edges based on the provided data. It calculates the minimum and maximum
    values for each dimension across all trajectories, and creates bins that
    span slightly beyond these extrema (by `pad` units on each side).

    If `lower_percentile` and `upper_percentile` are provided, the function uses
    these percentiles to determine the bin limits (i.e., uses the value at the
    specified percentiles as the min and max for binning, instead of the
    absolute min and max). If lower and upper percentiles are provided, the
    function ignores the `pad` parameter, since the bin limits are determined by
    the percentiles of the data rather than the absolute min and max.

    If `bin_limits` is provided, the function uses these user-defined limits to
    create the bins.

    If both `data` and `bin_limits` are provided, the function prioritizes
    `bin_limits` for bin creation.

    Parameters
    ----------
    bin_widths
        Tuple specifying the (approximate) bin width for each dimension.
    data
        Numpy array with shape (num_points, num_dimensions).
    bin_limits
        List of [min, max] pairs for each dimension specifying the bin limits.
    pad
        Amount to pad the automatically determined bin limits by on each side.
    lower_percentile
        Lower percentile to use when automatically determining bin limits from
        data.
    upper_percentile
        Upper percentile to use when automatically determining bin limits from
        data.

    Outputs: - bins: list of numpy arrays, each array contains
        the bin edges for a dimension
    - centers: list of numpy arrays, each array contains
        the center of each bin in a dimension

    If the dimension is 1, bins and centers are still lists (of length 1),
    containing the bin edges and centers for the single dimension.
    """
    if bin_limits is None and data is None:
        raise ValueError("Please provide data or upper and lower bounds for bins.")
    # need data to be shape (num_points, num_dimensions) for the rest of the
    # code to work, so reshape in the 1D case if necessary
    if data is not None and data.ndim == 1:
        data = data.reshape(-1, 1)
    ndim = data.shape[1] if data is not None else len(bin_limits)
    if ndim != len(bin_widths):
        raise ValueError(
            f"Mismatch between expected number of dimensions {ndim} and length of bin_widths {len(bin_widths)}."
        )
    bin_limits_: list[tuple[float, float]] = [] if bin_limits is None else bin_limits.copy()

    # Automatically determine bins based on data if bin limits are not provided
    if bin_limits is None:
        for i in range(ndim):
            # Get bin limits for this dimension across all trajectories either
            # based on absolute min and max (plus padding) or based on specified
            # percentiles
            if lower_percentile is not None:
                bin_min = np.percentile(data[:, i], lower_percentile)
            else:
                bin_min = data[:, i].min() - pad
            if upper_percentile is not None:
                bin_max = np.percentile(data[:, i], upper_percentile)
            else:
                bin_max = data[:, i].max() + pad
            bin_limits_.append((bin_min, bin_max))

    # Generate bins based on bin limits
    bins = []
    centers = []
    for i in range(ndim):
        # get number of bins for this dimension based on bin width
        num_bins = int(np.ceil((bin_limits_[i][1] - bin_limits_[i][0]) / bin_widths[i]))
        my_bins = np.linspace(bin_limits_[i][0], bin_limits_[i][1], num_bins + 1)
        bins.append(my_bins)
        centers.append(0.5 * (my_bins[1:] + my_bins[:-1]))

    bin_width_str = ", ".join([f"{bins[i][1] - bins[i][0]:.3f}" for i in range(len(bins))])
    logger.debug(
        "Generated bins for histogramming with bin widths: [ %s ]",
        bin_width_str,
    )
    return bins, centers


def _get_histogram_by_component_one_dataset(
    df: pd.DataFrame, bin_edges: list[np.ndarray], feat_cols: list[str] | None = None
) -> tuple[list[np.ndarray], pd.DataFrame]:
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
    num_frames = df[Column.TIMEPOINT].nunique()

    hist_array_list: list[np.ndarray] = [
        np.zeros((len(bin_edges[dim]) - 1, num_frames)) for dim in range(num_feats)
    ]  # histogram values for each component as a function of time

    # sort by timepoint
    df = df.sort_values(by=Column.TIMEPOINT).reset_index(drop=True)
    for t, df_frame in df.groupby(Column.TIMEPOINT):
        # loop over latent components
        for dim in range(num_feats):
            feats = df_frame[feat_cols[dim]].to_numpy()
            # compute histogram of feature data along each component
            t_index = df[Column.TIMEPOINT].unique().tolist().index(t)
            hist = np.histogram(feats, bins=bin_edges[dim], density=True)[0]
            hist_array_list[dim][:, t_index] = hist

            # update the dataframe with column of what bin
            # each crop at frame number t is in
            # along the given latent dimension
            # get the bin index for each crop
            bin_idx = np.digitize(feats, bin_edges[dim]) - 1
            # add the bin index to the dataframe (astype int)
            # restrict to crops at frame number t
            df.loc[df[Column.TIMEPOINT] == t, f"bin_{dim}"] = bin_idx

    # enforce that bin indices are integers
    # this is important for indexing later
    for dim in range(num_feats):
        df[f"bin_{dim}"] = df[f"bin_{dim}"].astype(int)

    # return the histogram array and the updated dataframe
    return hist_array_list, df


def get_histogram_by_component(
    df: pd.DataFrame,
    bin_width: float,
    bin_limits: list[tuple[float, float]],
    feat_cols: list[str] | None = None,
) -> tuple[list[list[np.ndarray]], list[np.ndarray], pd.DataFrame]:
    """
    Get histogram of feature data at each timepoint for each latent component
    across all datasets in the input dataframe.

    Input:
    - df: pd.DataFrame, feature data for multiple datasets
    - bin_width: float, width of each histogram bin
    - bin_limits: bin limits for each component
    - feat_cols: list[str] | None, column names of the features to use
    """
    # get column names for extracting feature data for a single dataset
    if feat_cols is None:
        # use all PCA feature columns in the dataframe
        feat_cols_all = DIFFAE_PC_COLUMN_NAMES
        feat_cols = [col for col in feat_cols_all if col in df.columns]

    num_feats = len(feat_cols)

    # check that bin_limits is provided and matches the number of features
    if len(bin_limits) != num_feats:
        raise ValueError(
            f"Number of bin limits ({len(bin_limits)}) must match number of features ({num_feats})"
        )

    # get bin edges for each feature dimension
    bin_edges = [
        get_bins([bin_width], bin_limits=[bin_limits[dim]])[0][0] for dim in range(num_feats)
    ]

    # loop over each dataset in the dataframe
    # get histogram / bin indices for each dataset
    hist_array_list_all_datasets = []
    df_list = []
    for _, df_group in df.groupby(Column.DATASET):
        hist_array_list_one_dataset, df_group_ = _get_histogram_by_component_one_dataset(
            df_group, bin_edges, feat_cols
        )
        df_list.append(df_group_)
        hist_array_list_all_datasets.append(hist_array_list_one_dataset)

    df_all_datasets_binned = pd.concat(df_list, ignore_index=True)

    return hist_array_list_all_datasets, bin_edges, df_all_datasets_binned


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
