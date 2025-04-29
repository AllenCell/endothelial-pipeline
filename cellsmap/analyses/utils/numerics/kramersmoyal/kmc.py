import numpy as np
from scipy.signal import convolve
from scipy.special import factorial
from itertools import product
import inspect
from typing import Callable

from cellsmap.analyses.utils.numerics.kramersmoyal.binning import histogramdd
from cellsmap.analyses.utils.numerics.kramersmoyal import kernels

def string_to_kernel(kernel: str) -> Callable:
    '''
    Function to convert a string to the corresponding kernel function.

    Input:
    - kernel: string, name of the kernel function

    Output:
    - kernel_func: callable, the kernel function with the given name
    as defined in cellsmap.analyses.utils.numerics.kramersmoyal.kernels
    '''
    # get dictionary of all callable functions in the kernels module
    not_kernel = {'factorial2','kernel','wraps','silvermans_rule'} # functions that are not kernels
    kernel_dict = {
        name: func for name, func in inspect.getmembers(kernels, inspect.isfunction) if name not in not_kernel
    }
    if kernel in kernel_dict:
        return kernel_dict[kernel]
    else:
        raise ValueError(f"Kernel '{kernel}' not recognized. Available kernels: {list(kernel_dict.keys())}")

def km(timeseries: list[np.ndarray]|np.ndarray, 
        grads:list[np.ndarray]|np.ndarray|None = None,
        bins: str='default', 
        powers: int=4,
        kernel: str='epanechnikov', 
        bw: float|None=None, 
        tol: float=1e-3,
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

        The order that they appear dictactes
        the order in the output `kmc`.

    kernel: string (default `epanechnikov`)
        Kernel used to convolute with the Kramers-Moyal coefficients. To select
        for example a Gaussian kernel use
            `kernel = `gaussian`
        Has to be a kernel implemented in `cellsmap.analyses.utils.numerics.kramersmoyal.kernels`.

    bw: float (default `None`)
        Desired bandwidth of the kernel. A value of 1 occupies the full space of
        the bin space. Recommended are values `0.005 < bw < 0.5`. 

    tol: float (default `1e-10`)
        Round to zero absolute values smaller than `tol`, after the
        convolutions.

    conv_method: str (default `auto`)
        A string indicating which method to use to calculate the convolution.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html

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
    .. [Gorjão2019] L. R. Gorjão and F. Meirinhos, "kramersmoyal: Kramers-Moyal
    coefficients for stochastic processes." Journal of Open Source Software 4(44), 
    1693, 2019. DOI: 10.21105/joss.01693
    """
    # check inputs (case of multi_traj and single traj)
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

    assert dims == len(bins), "bins not matching timeseries' dimension"

    # convert specified kernel to callable
    kernel_func = string_to_kernel(kernel)

    # check bandwidth input
    if bw is None:
        bw = kernels.silvermans_rule(timeseries,multi_traj=multi_traj)
    elif callable(bw):
        bw = bw(timeseries)
    assert bw > 0.0, "Bandwidth must be > 0"

    print(f"Using bandwidth {bw} for kernel {kernel}.")

    # This is where the calculations take place
    kmc = _km(timeseries, grads, bins, powers, kernel_func, 
                      bw, tol, conv_method, multi_traj)

    return kmc


def _km(timeseries: list[np.ndarray]|np.ndarray,
        grads: list[np.ndarray]|np.ndarray|None, 
        bins: np.ndarray, powers: np.ndarray,
        kernel_func: callable, bw: float, tol: float,
        conv_method: str, multi_traj:bool) -> np.ndarray:
    """
    Helper function for `km` that does the heavy lifting and actually estimates
    the Kramers─Moyal coefficients from the timeseries.
    """
    # Internal function to get the Cartesian product of the bin edges
    def cartesian_product(arrays: np.ndarray):
        # Taken from https://stackoverflow.com/questions/11144513
        la = len(arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=np.float64)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr

    ##### Calculate derivative (if not provided)
    if grads is None:
        # Calculate the gradients if not provided
        if multi_traj:
            grads = [np.diff(ts, axis=0) for ts in timeseries]
        else:
            grads = np.diff(timeseries, axis=0)
    # Check if the gradients are in the right shape, trim timeseries to the same length
    if multi_traj:
        # Concatenate all gradients, need to get weights for weighted histogram
        grads = np.concatenate(grads, axis=0)
        # Get trajectories for weighted histogram (timepoints corresponding to the gradients)
        timeseries_ = np.concatenate([ts[:-1] for ts in timeseries], axis=0)
    else:
        timeseries_ = timeseries[:-1]

    ##### Weights: for each displacement vector, get the coresponding powers/products
    #               of the gradients for the Kramers─Moyal coefficients.
        
    # Raises each component of the gradient array to the corresponding 
    #    element of the powers and then multiplies them together.
    # e.g., for 2D, powers = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]], we have:
    # > np.power(grads.T, powers[..., None]) = [[1, 1], 
    #                                           [x_0(t+1)-x_0(t), 1] 
    #                                           [1 , x_1(t+1)-x_1(t)], 
    #                                           [x_0(t+1)-x_0(t), (x_1(t+1)-x_1(t))],
    #                                           [(x_0(t+1)-x_0(t))^2, 1], 
    #                                           [1, (x_1(t+1)-x_1(t))^2]]
    # > np.prod(..., axis=1) = [1, 
    #                           (x_0(t+1)-x_0(t)), 
    #                           (x_1(t+1)-x_1(t)),
    #                           (x_0(t+1)-x_0(t))(x_1(t+1)-x_1(t)),
    #                           (x_0(t+1)-x_0(t))^2, 
    #                           (x_1(t+1)-x_1(t))^2]
    # If there are L powers and M observations, the result is an L x M array.
    weights = np.prod(np.power(grads.T, powers[..., None]), axis=1)

    ##### Get weighted histogram for convolution

    # If there are L powers, the result in an L x N[0] x N[1] x ... x N[D-1] array
    # where N[i] is the number of bins in dimension i.
    hist, edges = histogramdd(timeseries_, bins=bins,
                              weights=weights, bw=bw)
    

    ##### Generate centered kernel on larger grid (fft'ed convolutions are circular).

    # Map edges to interval [0, L_i] in each dimension, where L_i is the number of bins
    # times the bin width in dimension i. Then edges_k is the bin edges
    # for the interval [-L_i, L_i] with the same bin width. This grid is twice the size
    # of the histogram in each dimension and centered around the origin.

    # The kernel is then evaluated at all points in this extended grid (obtained 
    # via the cartesian product of the entries of edges_k).
    # The purpose of this is to artifically construct a periodic kernel
    # that is centered around the origin, so that the input into the convolution
    # is compatible with the circular nature of the convolution obtained via fft.
    # (Default convolution method is 'auto', which uses fft if the kernel is large enough.)
    edges_k = [(e[1] - e[0]) * np.arange(-e.size, e.size + 1) for e in edges]
    kernel_ = kernel_func(cartesian_product(edges_k), bw=bw)


    ##### KMC computation: convolve the histogram with the kernel

    # Convolve weighted histogram of kmc observations (displacements ^ powers) 
    # with augmented periodic kernel and trim it back to the original size.
    # Note that the first entry of the output is the normalization factor
    # of the kernel estimator, which is the same for all dimensions.
    # The normalization factor is the KDE of the empirical density function.
    kmc = convolve(hist, kernel_[None, ...], mode='same', method=conv_method)

    # Normalise with correct factorial coefficients * histogram
    mask = np.abs(kmc[0]) < tol # where probability density is small... (i.e., little to no data)
    kmc[0:, mask] = np.nan # ...set kmc coeffs to nan

    # get correct Taylor expansion coefficients (e.g., divide 2nd order powers by 2!)
    taylors = np.prod(factorial(powers[1:]), axis=1)
    kmc[1:, ~mask] /= taylors[..., None] * kmc[0, ~mask] # divide by Taylor coeff * 0th order coeffs (probability density)

    return kmc
