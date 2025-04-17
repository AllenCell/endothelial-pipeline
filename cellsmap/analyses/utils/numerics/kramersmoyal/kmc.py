import numpy as np
from scipy.signal import convolve
from scipy.special import factorial
from itertools import product
from typing import Tuple

from .binning import histogramdd
from .kernels import silvermans_rule, epanechnikov

def km(timeseries: list[np.ndarray]|np.ndarray, 
        grads:list[np.ndarray]|np.ndarray|None = None,
        bins: str='default', powers: int=4,
        kernel: callable=epanechnikov, bw: float=None, tol: float=1e-10,
        multi_traj: bool=False,
        conv_method: str='auto') -> np.ndarray:
    """
    Estimates the Kramers─Moyal coefficients from a timeseries using a kernel
    estimator method. `km` can calculate the Kramers─Moyal coefficients for a
    timeseries of any dimension, up to any desired power.

    Parameters
    ----------
    timeseries: list of np.ndarrays (if multi_traj is True) or np.ndarray
        The D-dimensional timeseries `(N, D)`. The timeseries of length `N`
        and dimensions `D`.

    grads: list of np.ndarrays (if multi_traj is True) or np.ndarray
        The displacement vectors of the timeseries. The gradients of length `N` and
        dimensions `D`.

    bins: int or list or np.ndarray or string (default `default`)
        The number of bins. This is the underlying space for the Kramers─Moyal
        coefficients to be estimated. If desired, bins along each dimension can
        be given as monotonically increasing bin edges (tuple or list), e.g.,

        * in 1-D, `(np.linspace(lower, upper, length),)`;
        * in 2-D, `(np.linspace(lower_x, upper_x, length_x),
                    np.linspace(lower_y, upper_y, length_y))`,

        with desired `lower` and `upper` ranges (in each dimension).
        If default, the bin numbers for different dimensions are:

        * 1-D, 5000;
        * 2-D, 100×100;
        * 3-D, 25×25×25.

        The bumber of bins along each dimension can be specified, e.g.,

        * 2-D, `[125, 75]`,
        * 3-D, `[100, 80, 120]`.

        If `bins` is int, or a list or np.array of dimension 1, and the
        `timeseries` dimension is `D`, then `int(bins**(1/D))`.

    powers: int or list or tuple or np.ndarray (default `4`)
        Powers for the operation of calculating the Kramers─Moyal coefficients.
        Default is the largest power used, e.g., if `4`, then `(0, 1, 2, 3, 4)`.
        They can be specified, matching the dimensions of the timeseries. E.g.,
        in 1-dimension the first four Kramers─Moyal coefficients can be given as
        `powers=(0, 1, 2, 3, 4)`, which is the same as `powers=4`. Setting
        `powers=p` for higher dimensions will results in all possible
        combinations up to the desired power 'p', e.g.

        * 2-D, `powers=2` results in
            powers = np.array([[0, 0, 1, 1, 0, 1, 2, 2, 2],
                               [0, 1, 0, 1, 2, 2, 0, 1, 2]]).T

        Set `full=True` to output `powers`. The order that they appear dictactes
        the order in the output `kmc`.

    kernel: callable (default `epanechnikov`)
        Kernel used to convolute with the Kramers-Moyal coefficients. To select
        for example a Gaussian kernel use

            `kernel = kernels.gaussian`

    bw: float (default `None`)
        Desired bandwidth of the kernel. A value of 1 occupies the full space of
        the bin space. Recommended are values `0.005 < bw < 0.5`. Set
        `full=True` to output `bw`.

    tol: float (default `1e-10`)
        Round to zero absolute values smaller than `tol`, after the
        convolutions.

    conv_method: str (default `auto`)
        A string indicating which method to use to calculate the convolution.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html

    center_edges: bool (default `True`)
        Whether to return the bin centers or the bin edges (since for `n` bins
        there are `n + 1` edges).

    full: bool (default `False`)
        Outputs bandwidth `bw` and powers `powers`.

    Returns
    -------
    kmc: np.ndarray
        The calculated Kramers─Moyal coefficients in accordance to the
        timeseries dimensions in `(D, bins.shape)` shape. To extract the
        selected orders of the kmc, use `kmc[i,...]`, with `i` the order
        according to powers.

    edges: np.ndarray
        The bin edges with shape `(D, bins.shape)` of the estimated
        Kramers─Moyal coefficients.

    (..., bw, powers): tuple
        This is only returned if `full=True`:

        * The bandwidth `bw`,
        * An array of the `powers`.

    References
    ----------
    .. [Lamouroux2009] D. Lamouroux and K. Lehnertz, "Kernel-based regression of
    drift and diffusion coefficients of stochastic processes." Physics Letters A
    373(39), 3507─3512, 2009. DOI: 10.1016/j.physleta.2009.07.073
    """
    if multi_traj:
        assert len(timeseries) == len(grads), \
            "Must have gradients for each timeseries"
        assert len(timeseries) > 0, "No data in timeseries"
        assert len(grads) > 0, "No data in gradients"
        assert len(grads[0]) == len(timeseries[0])-1, \
            "Need to have gradients for each timepoint in timeseries except for last"
        timeseries = [np.asarray_chkfinite(ts, dtype=float) for ts in timeseries]
        grads = [np.asarray_chkfinite(g, dtype=float) for g in grads]

        for j, ts in enumerate(timeseries):
            if len(ts.shape) == 1:
                timeseries[j] = ts.reshape(-1, 1)

        m = len(timeseries)
        dims = timeseries[0].shape[1]
    else:
        # Check finiteness, dimensions, and existence of the time series
        timeseries = np.asarray_chkfinite(timeseries, dtype=float)
        grads = np.asarray_chkfinite(grads, dtype=float)
        if len(timeseries.shape) == 1:
            timeseries = timeseries.reshape(-1, 1)
        if len(grads.shape) == 1:
            grads = grads.reshape(-1, 1)

        assert len(timeseries.shape) == 2, "Timeseries must be (N, D) shape"
        assert timeseries.shape[0] > 0, "No data in timeseries"

        dims = timeseries.shape[1]

    # Tranforming powers into right shape
    if isinstance(powers, int):
        # complicated way of obtaing powers in all dimensions
        powers = np.array(sorted(product(*(range(powers + 1),) * dims),
            key=lambda x: (max(x), x)))

    powers = np.asarray_chkfinite(powers, dtype=float)
    if len(powers.shape) == 1:
        powers = powers.reshape(-1, 1)

    if not (powers[0] == [0] * dims).all():
        powers = np.array([[0] * dims, *powers])

    assert (powers[0] == [0] * dims).all(), "First power must be zero"
    assert dims == powers.shape[1], "Powers not matching timeseries' dimension"

    # Check and adjust bins
    if isinstance(bins, str):
        if bins == 'default':
            bins = [5000] if dims == 1 else bins
            bins = [100] * 2 if dims == 2 else bins
            bins = [25] * 3 if dims == 3 else bins
        assert dims < 4, "If dimension of timeseries > 3, set bins manually"

    if isinstance(bins, int):
        bins = [int(bins**(1/dims))] * dims

    if isinstance(bins, (list, tuple)):
        assert all(isinstance(ele, (int, np.ndarray)) for ele in bins), \
            "list or tuples of bins must either be ints or arrays"

    # bins = np.asarray_chkfinite(bins, dtype=int)
    assert dims == len(bins), "bins not matching timeseries' dimension"

    if bw is None:
        bw = silvermans_rule(timeseries)
    elif callable(bw):
        bw = bw(timeseries)
    assert bw > 0.0, "Bandwidth must be > 0"

    # This is where the calculations take place
    kmc = _km(timeseries, grads, bins, powers, kernel, 
                      bw, tol, conv_method, multi_traj)

    return kmc



def _km(timeseries: list[np.ndarray]|np.ndarray,
        grads: list[np.ndarray]|np.ndarray|None, 
        bins: np.ndarray, powers: np.ndarray,
        kernel: callable, bw: float, tol: float,
        conv_method: str, multi_traj:bool) -> np.ndarray:
    """
    Helper function for `km` that does the heavy lifting and actually estimates
    the Kramers─Moyal coefficients from the timeseries.
    """
    def cartesian_product(arrays: np.ndarray):
        # Taken from https://stackoverflow.com/questions/11144513
        la = len(arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=np.float64)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr

    # Calculate derivative and the product of its powers
    dims = len(bins)
    if grads is None:
        # Calculate the gradients if not provided
        if multi_traj:
            grads = [np.diff(ts, axis=0) for ts in timeseries]
        else:
            grads = np.diff(timeseries, axis=0)

    if multi_traj:
        # Concatenate all gradients, need to get weights for weighted histogram
        grads = np.concatenate(grads, axis=0)
        # Get trajectories for weighted histogram (timepoints corresponding to the gradients)
        timeseries_ = np.concatenate([ts[:-1] for ts in timeseries], axis=0)
        assert grads.shape[1] == dims
    else:
        timeseries_ = timeseries[:-1]

    weights = np.prod(np.power(grads.T, powers[..., None]), axis=1)

    # Get weighted histogram
    hist, edges = histogramdd(timeseries_, bins=bins,
                              weights=weights, bw=bw)

    # Generate centred kernel on larger grid (fft'ed convolutions are circular)
    edges_k = [(e[1] - e[0]) * np.arange(-e.size, e.size + 1) for e in edges]
    kernel_ = kernel(cartesian_product(edges_k), bw=bw)

    # Convolve weighted histogram with kernel and trim it
    kmc = convolve(hist, kernel_[None, ...], mode='same', method=conv_method)

    # Normalise
    mask = np.abs(kmc[0]) < tol
    kmc[0:, mask] = 0.0
    taylors = np.prod(factorial(powers[1:]), axis=1)
    kmc[1:, ~mask] /= taylors[..., None] * kmc[0, ~mask]

    return kmc
