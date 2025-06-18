import numpy as np


def _bincount(x: np.ndarray, weights: np.ndarray, minlength: int = 0):
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
    n_count = tuple(
        np.searchsorted(edges[i], sample[:, i], side="right") for i in range(d)
    )

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
    xy = np.ravel_multi_index(n_count, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = _bincount(xy, weights, minlength=nbin.prod())
    return hist


def histogramdd(
    sample: np.ndarray, bins: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
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
    dedges = d * [None]
    nbin = np.zeros(d, dtype=int)
    weights = np.asarray(weights)
    for i in range(d):
        nbin[i] = len(edges[i]) + 1
        # check that bins are monotonically increasing
        if np.any(edges[i][:-1] > edges[i][1:]):
            raise ValueError(
                f"`bins[{i}]` must be monotonically increasing, when an array"
            )
        # increase bin count by 1 to include outliers
        nbin[i] = len(edges[i]) + 1
        # get the width of each bin
        dedges[i] = np.diff(edges[i])

    m = len(bins)
    if m != d:
        raise ValueError(
            "The dimension of bins must be equal to the dimension of the " " sample x"
        )

    # Get the histogram counts.
    hist = _get_bin_counts(sample, weights, edges, d, nbin)

    # Reshape the histogram matrix to the correct shape.
    if weights.ndim == 1:
        hist = hist.reshape(nbin)
    else:
        hist = hist.reshape((weights.shape[0], *nbin))

    # Remove outliers (indices 0 and -1 for each dimension).
    core = d * (slice(1, -1),)
    hist = hist[(..., *core)]

    return hist
