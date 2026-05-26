"""Methods related to binning and histogram calculations."""

import logging

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
    """Compute the q-th percentile of circular data.

    Parameters
    ----------
    angles
        1D array of circular data (e.g., angles in radians).
    q
        Percentile to compute (between 0 and 100).
    polar_range
        Tuple specifying the circular range of the data (e.g., (0,
        np.pi) for angles in radians).

    Returns
    -------
    :
        The q-th percentile of the circular data, wrapped to the specified polar range.

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
    """Generate histogram bins either automatically based on data or user-defined bin limits.

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

    Returns
    -------
    :
        List of numpy arrays containing the bin edges for each dimension.
    :
        List of numpy arrays containing the bin centers for each dimension.

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
) -> list[np.ndarray]:
    """Compute histogram of feature data at each timepoint for each latent component.

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
        for dim, feat_col in enumerate(feat_cols):
            feats = df_frame[feat_col].to_numpy()
            # compute histogram of feature data along each component
            t_index = df[Column.TIMEPOINT].unique().tolist().index(t)
            hist = np.histogram(feats, bins=bin_edges[dim], density=True)[0]
            hist_array_list[dim][:, t_index] = hist

    return hist_array_list


def get_normalization_constant(p: np.ndarray, dx: list) -> np.ndarray:
    """Get normalization constant for probability distribution p.

    The normalization constant is the integral of the probability distribution
    over the state space.

    Parameters
    ----------
    p
        Probability distribution to normalize, defined on a grid. The shape of p
        should be N[1] x N[2] x ... x N[ndim], where N[i] is the number of bins
        in the i-th dimension.
    dx
        List of bin widths in each dimension, used for numerical integration.

    Returns
    -------
    :
        Normalization constant for the probability distribution p.

    """
    ndim = len(dx)  # number of dimensions

    # copy p to avoid modifying the original array
    c = p.copy()
    for i in range(ndim):
        # integrate over axis=0 as we marginalize over each dimension
        c = np.trapz(c, dx=dx[i], axis=0)

    return c


def _get_bin_counts(
    sample: np.ndarray,
    weights: np.ndarray,
    edges: list[np.ndarray],
    nbin: np.ndarray,
) -> np.ndarray:
    """Get weighted bin counts for the input sample.

    This function computes the bin number each sample falls into using
    `numpy.searchsorted`, and then uses `numpy.bincount` to count the number of
    samples in each bin, weighted by the provided weights.

    Parameters
    ----------
    sample
        The data to be histogrammed, with shape (n_samples, n_dimensions).
    weights
        An array of weights for each sample, with shape (n_samples,) or (n_weights, n_samples).
    edges
        A list of 1D arrays specifying the bin edges for each dimension.
    nbin
        An array specifying the number of bins in each dimension, including outliers.

    Returns
    -------
    :
        An array of weighted bin counts for the input sample, with shape (n_weights, n_bins).

    """
    num_dims = sample.shape[1]

    # Compute the bin number each sample falls into.
    n_count = tuple(np.searchsorted(edges[i], sample[:, i], side="right") for i in range(num_dims))

    # Adjust behavior of searchsorted for samples that fall on the rightmost
    # edge of the last bin.
    for i in range(num_dims):
        # Find which points are on the rightmost edge.
        on_edge = sample[:, i] == edges[i][-1]
        # Shift these points one bin to the left.
        n_count[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # Ensure n_count is a tuple of integer arrays
    n_count = tuple(arr.astype(int) for arr in n_count)
    xy = np.ravel_multi_index(n_count, tuple(map(int, nbin)))

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = np.array([np.bincount(xy, w, minlength=int(np.prod(nbin))) for w in weights])
    return hist


def histogramdd(sample: np.ndarray, bins: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Compute the multidimensional weighted histogram of a sample.

    Allows for a weights matrix to be passed in, which is used to weight the
    samples in each bin.

    This code is a modified version of the histogramdd function in `numpy`, with
    the addition of a weights matrix.

    Part of the following code is licensed under the BSD-3 License (from `numpy`).

    Parameters
    ----------
    sample
        The data to be histogrammed, with shape (n_samples, n_dimensions).
    bins
        A list of 1D arrays specifying the bin edges for each dimension.
    weights
        An array of weights for each sample, with shape (n_samples,) or (n_weights, n_samples).

    Returns
    -------
    :
        An array of weighted bin counts for the input sample, with shape (n_weights, n_bins).

    """
    num_dims = sample.shape[-1]
    # initialize edges, dedges, and nbin
    edges = bins.copy()
    dedges = []
    nbin = np.zeros(num_dims, dtype=int)
    weights = np.asarray(weights)
    for i in range(num_dims):
        nbin[i] = len(edges[i]) + 1
        # check that bins are monotonically increasing
        if np.any(edges[i][:-1] > edges[i][1:]):
            raise ValueError(f"`bins[{i}]` must be monotonically increasing, when an array")
        # increase bin count by 1 to include outliers
        nbin[i] = len(edges[i]) + 1
        # get the width of each bin
        dedges.append(np.diff(edges[i]))

    if len(bins) != num_dims:
        raise ValueError("The dimension of bins must be equal to the dimension of the " " sample x")

    # Get the histogram counts.
    hist: np.ndarray = _get_bin_counts(sample, weights, edges, nbin)

    # Reshape the histogram matrix to the correct shape.
    if weights.ndim == 1:
        hist = hist.reshape(nbin)
    else:
        hist = hist.reshape((weights.shape[0], *nbin))

    # Remove outliers (indices 0 and -1 for each dimension).
    core: tuple[slice, ...] = num_dims * (slice(1, -1),)

    # slice the histogram to remove outliers
    hist = hist[..., *core]

    return hist


def adjust_limits_from_bin_size(
    data_min_max: tuple[float, float],
    defined_min_max: tuple[float | None, float | None],
    bin_size: float,
) -> tuple[float, float]:
    """Adjust some (min, max) limits based on the data limits, bin size, and defined limits such
    that the limits can fit a whole number of bins.
    If the defined limits are not None, then use those. Otherwise, adjust the limits based on the
    data limits and bin size.

    Parameters
    ----------
    data_min_max
        Tuple of (min, max) values for the data to be binned.
    defined_min_max
        Tuple of (min, max) values for the bin limits defined by the user. If None,
        the limits will be adjusted based on the data limits and bin size.
    bin_size
        Size of the bins to be used for binning the data.
    """
    data_min, data_max = data_min_max
    defined_min, defined_max = defined_min_max

    if defined_min is not None and data_min < defined_min:
        raise ValueError("Minimum bin value from data is less than defined bin minimum.")
    if defined_max is not None and data_max > defined_max:
        raise ValueError("Maximum bin value from data is greater than defined bin maximum.")

    adjust_lim_min_from_bin_func = lambda bin_min, bin_min_lim: (
        np.floor(bin_min / bin_size) * bin_size if bin_min_lim is None else bin_min_lim
    )
    adjust_lim_max_from_bin_func = lambda bin_max, bin_max_lim: (
        np.ceil(bin_max / bin_size) * bin_size if bin_max_lim is None else bin_max_lim
    )
    bin_min = adjust_lim_min_from_bin_func(data_min, defined_min)
    bin_max = adjust_lim_max_from_bin_func(data_max, defined_max)

    return bin_min, bin_max
