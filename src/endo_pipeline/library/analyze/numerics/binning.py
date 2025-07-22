import numpy as np


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
