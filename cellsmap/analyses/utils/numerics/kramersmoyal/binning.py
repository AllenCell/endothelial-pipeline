import numpy as np


def _get_outer_edges(
    a: np.ndarray, edge_range: tuple | list, bw: float
) -> tuple[float, float]:
    """
    Determine the outer bin edges to use, from either the data or the range
    argument.
    """
    if edge_range is not None:
        first_edge, last_edge = edge_range
        if first_edge > last_edge:
            raise ValueError("Max must be larger than min in range parameter")
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"Supplied range of [{first_edge}, {last_edge}] " " is not finite"
            )
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min() - bw, a.max() + bw
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"Autodetected range of [{first_edge}, {last_edge}] " " is not finite"
            )

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def histogramdd(
    sample: np.ndarray,
    bins: int = 10,
    edge_range: tuple | list | None = None,
    density: bool = True,
    weights: np.ndarray | None = None,
    bw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the multidimensional histogram of a sample.
    An alternative to Numpy's histogramdd, supporting a weights matrix.
    Part of the following code is licensed under the BSD-3 License (from Numpy).
    """

    try:
        # Sample is an ND-array.
        n, d = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        n, d = sample.shape

    nbin = np.empty(d, int)
    edges = d * [None]
    dedges = d * [None]
    if weights is not None:
        weights = np.asarray(weights)

    try:
        m = len(bins)
        if m != d:
            raise ValueError(
                "The dimension of bins must be equal to the dimension of the "
                " sample x"
            )
    except TypeError:
        # bins is an integer
        bins = d * [bins]

    # normalize the range argument
    if edge_range is None:
        edge_range = (None,) * d
    elif len(edge_range) != d:
        raise ValueError("Range argument must have one entry per dimension")

    # Create edge arrays
    for i in range(d):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(f"`bins[{i}]` must be positive, when an integer")
            smin, smax = _get_outer_edges(sample[:, i], edge_range[i], bw)
            edges[i] = np.linspace(smin, smax, bins[i] + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    f"`bins[{i}]` must be monotonically increasing, when an array"
                )
        else:
            raise ValueError(f"`bins[{i}]` must be a scalar or 1d array")

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side="right")
        for i in range(d)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(d):
        # Find which points are on the rightmost edge.
        on_edge = sample[:, i] == edges[i][-1]
        # Shift these points one bin to the left.
        ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = bincount1(xy, weights, minlength=nbin.prod())

    # Shape into a proper matrix
    if weights.ndim == 1:
        hist = hist.reshape(nbin)
    else:
        hist = hist.reshape((weights.shape[0], *nbin))

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting="safe")

    # Remove outliers (indices 0 and -1 for each dimension).
    core = d * (slice(1, -1),)
    hist = hist[(..., *core)]

    # if density is requested, normalize the histogram
    if density:
        if weights.ndim == 1:
            # calculate the probability density function
            s = hist.sum()
            for i in range(d):
                shape = np.ones(d, int)
                shape[i] = nbin[i] - 2
                hist = hist / dedges[i].reshape(shape)
            hist /= s
        else:
            for dd in range(weights.shape[0]):
                s = hist[dd, ...].sum()
                for i in range(d):
                    shape = np.ones(d, int)
                    shape[i] = nbin[i] - 2
                    hist[dd, ...] = hist[dd, ...] / dedges[i].reshape(shape)
                hist[dd, ...] /= s

    if weights.ndim == 1:
        if (hist.shape != nbin - 2).any():
            raise RuntimeError("Internal Shape Error")
    else:
        if (hist.shape != np.array([weights.shape[0], *(nbin - 2)])).any():
            raise RuntimeError("Internal Shape Error")
    return hist, edges


def bincount1(x, weights, minlength=0):
    """
    Wrap the function np.bincount in a way
      that handles weights.
    """
    return np.array([np.bincount(x, w, minlength=minlength) for w in weights])
