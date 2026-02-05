import inspect
from collections.abc import Callable

import numpy as np
from scipy.signal import convolve
from scipy.special import factorial

from endo_pipeline.library.analyze.numerics import histogramdd

from . import km_kernels


def _string_to_kernel(kernel: str) -> Callable:
    """
    Convert a string to the corresponding kernel function.

    Input:
    - kernel: string, name of the kernel function

    Output:
    - kernel_func: callable, the kernel function with the given name
    as defined in km_kernels
    """
    # get dictionary of all callable functions in the kernels module
    not_kernel = {
        "factorial2",
        "kernel",
        "wraps",
        "silvermans_rule",
    }  # functions that are not kernels
    kernel_dict = {
        name: func
        for name, func in inspect.getmembers(km_kernels, inspect.isfunction)
        if name not in not_kernel
    }
    if kernel in kernel_dict:
        return kernel_dict[kernel]
    else:
        raise ValueError(
            f"Kernel '{kernel}' not recognized. " f" Available kernels: {list(kernel_dict.keys())}"
        )


def _km_wrapper(
    timeseries: list[np.ndarray],
    grads: list[np.ndarray],
    bins: list[np.ndarray],
    powers: np.ndarray,
    kernel: str,
    bw: float,
    tol: float = 1e-10,
    conv_method: str = "auto",
) -> np.ndarray:
    """
    Estimates the Kramers─Moyal coefficients from a timeseries using a kernel
    estimator method. `km` can calculate the Kramers─Moyal coefficients for a
    timeseries of any dimension, up to any desired power.

    Parameters
    ----------
    timeseries: list of np.ndarrays
        The set of d-dimensional timeseries `(n, d)`, where
        n is the number of timepoints and d is the dimension.

    grads: list of np.ndarrays
        The displacement vectors of the timeseries.
        (length `n-1` and dimensions `d`).
        The number of trajectories in `timeseries` must
        be equal to the number of gradients in `grads`.

    bins: list of np.ndarrays
        List of monotonically increasing bin edges in each dimension.
        This is the underlying space for the Kramers─Moyal
        coefficients to be estimated.

    powers: np.ndarray
        Powers for the operation of calculating the Kramers─Moyal coefficients.
        The powers are the exponents of the components of the displacement
        vectors in the Kramers─Moyal coefficients.

        The powers are given in the form of a 2-D array,
        where each row corresponds to a power, and each column
        corresponds to a component of the displacement vector
        that is raised to that power. The first row is always
        zero, to account for the normalization of the coefficients.

        * e.g., to compute each component of the drift
        coefficient in 2D, the powers are `[[0, 0], [1, 0], [0, 1]]`,
        where the first row is the normalization

        The powers can be computed by calling `_get_km_powers(ndim)`, where
        `ndim` is the dimension of the timeseries. The powers are then
        automatically generated, but only up to second order.

        The order that they appear dictates the order of the
        corresponding coefficient in the output `kmc`.

    kernel: string
        Kernel used to convolute with the Kramers-Moyal coefficients.
        To select, for example, a Gaussian kernel use
            `kernel = `gaussian`
        Has to be the name of a kernel implemented in
        `library.analyze.kramersmoyal.kernels`.

    bw: float (default `None`)
        Desired bandwidth of the kernel. A value of 1 occupies
        the full space of the bin space.
        Recommended are values `0.005 < bw < 0.5`.

    tol: float (default `1e-10`)
        Round to zero absolute values smaller than `tol`, after the
        convolutions. These points are set to `NaN` in the output.

    conv_method: str (default `auto`)
        A string indicating which method to use to calculate the convolution.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html

    Returns
    -------
    kmc: np.ndarray
        The calculated Kramers─Moyal coefficients in accordance to the
        timeseries dimensions in `(d, bins.shape)` shape. To extract the
        selected orders of the kmc, use `kmc[i,...]`, with `i` the order
        according to powers.

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
    assert len(timeseries) == len(grads), "Must have gradients for each timeseries"
    assert len(timeseries) > 0, "No data in timeseries"
    assert len(grads) > 0, "No data in gradients"
    assert (
        len(grads[0]) == len(timeseries[0]) - 1
    ), "Need to have gradients for each timepoint in timeseries except for last"
    timeseries = [np.asarray_chkfinite(ts, dtype=float) for ts in timeseries]
    grads = [np.asarray_chkfinite(g, dtype=float) for g in grads]

    for j, ts in enumerate(timeseries):
        if len(ts.shape) == 1:
            timeseries[j] = ts.reshape(-1, 1)

    # get dimension of the timeseries
    ndim = timeseries[0].shape[1]

    powers = np.asarray_chkfinite(powers, dtype=float)
    # check if powers is a 1D array
    # if so, reshape it to a 2D array with one column
    if len(powers.shape) == 1:
        powers = powers.reshape(-1, 1)

    # add normalization factor to powers
    # if the first row is not all zeros
    if not (powers[0] == [0] * ndim).all():
        powers = np.array([[0] * ndim, *powers])

    assert ndim == powers.shape[1], "Powers not matching timeseries' dimension"
    assert ndim == len(bins), "bins not matching timeseries' dimension"

    # convert specified kernel to callable
    kernel_func = _string_to_kernel(kernel)

    # This is where the calculations take place
    kmc = _km_worker(timeseries, grads, bins, powers, kernel_func, bw, tol, conv_method)

    return kmc


def get_cartesian_product(arrays: np.ndarray | list) -> np.ndarray:
    shapes = [len(arr) for arr in arrays]
    indices = np.indices(shapes, sparse=False)
    unstacked = [arrays[i][sub_indices] for i, sub_indices in enumerate(indices)]
    return np.stack(unstacked, axis=-1)


def _km_worker(
    timeseries: list[np.ndarray],
    grads: list[np.ndarray],
    bins: list[np.ndarray],
    powers: np.ndarray,
    kernel_func: Callable,
    bw: float,
    tol: float,
    conv_method: str,
) -> np.ndarray:
    """
    Estimate the Kramers─Moyal coefficients from a timeseries using a kernel
    estimator method. This is the internal function that does the heavy lifting.
    """

    # Concatenate all gradients, need to get weights for weighted histogram
    grads_concat: np.ndarray = np.concatenate(grads, axis=0)
    # Get trajectories for weighted histogram
    # (timepoints corresponding to the gradients)
    # Note that the last timepoint of each trajectory
    # is not included, as it is not used in the gradients.
    timeseries_ = np.concatenate([ts[:-1] for ts in timeseries], axis=0)

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
    weights = np.prod(np.power(grads_concat.T, powers[..., None]), axis=1)

    ##### Get weighted histogram for convolution

    # If there are L powers, the result in an L x N[0] x N[1] x ... x N[D-1] array
    # where N[i] is the number of bins in dimension i.
    hist = histogramdd(timeseries_, bins=bins, weights=weights)

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
    # (Default convolution method is 'auto', which uses
    # fft if the kernel is large enough.)
    edges_k = [(e[1] - e[0]) * np.arange(-e.size, e.size + 1) for e in bins]
    kernel_ = kernel_func(get_cartesian_product(edges_k), bw=bw)

    ##### KMC computation: convolve the histogram with the kernel

    # Convolve weighted histogram of kmc observations (displacements ^ powers)
    # with augmented periodic kernel and trim it back to the original size.
    # Note that the first entry of the output is the normalization factor
    # of the kernel estimator, which is the same for all dimensions.
    # The normalization factor is the KDE of the empirical density function.
    kmc = convolve(hist, kernel_[None, ...], mode="same", method=conv_method)

    # Normalise with correct factorial coefficients * histogram
    mask = np.abs(kmc[0]) < tol  # where probability density is small... (i.e., little to no data)
    kmc[0:, mask] = np.nan  # ...set kmc coeffs to nan

    # get correct Taylor expansion coefficients (e.g., divide 2nd order powers by 2!)
    taylors = np.prod(factorial(powers[1:]), axis=1)
    kmc[1:, ~mask] /= (
        taylors[..., None] * kmc[0, ~mask]
    )  # divide by Taylor coeff * 0th order coeffs (probability density)

    return kmc


def _get_km_powers(ndim: int) -> np.ndarray:
    """
    Generate the powers for the Kramers-Moyal coefficients
    based on the dimensionality of the data.

    Inputs:
    - ndim: number of dimensions in the data

    Outputs:
    - powers: numpy array of powers for Kramers-Moyal coefficients

    For example, for 1D data, the powers are:
    [[0],  # normalization for kernel convolution (density)
     [1],  # f
     [2]]  # D

    For 2D data, the powers are:
    [[0,0],  # normalization for kernel convolution (density)
     [1,0],  # f_1
     [0,1],  # f_2
     [2,0],  # D_1
     [0,2]]  # D_2
    """

    if ndim == 1:  # straightforward case for 1D data
        powers = np.array([[0], [1], [2]])
        #                   /    f    D
        #          index:   0    1    2
    else:  # if ndim > 1, utilize identity matrix to generate powers
        n_powers = 2 * ndim + 1
        powers = np.zeros((n_powers, ndim), dtype=int)  # row 0 is all zeros
        # drift powers: row 1 to ndim
        powers[1 : ndim + 1] = np.eye(ndim, dtype=int)
        # diffusion powers: row ndim+1 to end (no interaction terms)
        powers[ndim + 1 :] = 2 * np.eye(ndim, dtype=int)
    return powers


def _get_km_coeff(
    traj_list: list,
    d_traj_list: list,
    bins: list,
    dt: float,
    kernel_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get Kramers-Moyal coefficients for a list
    of trajectories in d-dimensional space using
    a kernel density estimation method.

    Inputs:
    - traj_list: list of numpy arrays, each array is a single
        trajectory in (d-dim) feature space
    - d_traj_list: list of numpy arrays, each array is the
        displacement vectors along that trajectory
    - bins: list of numpy arrays, each array contains the
        bin edges for a dimension (used for computing
        conditional averages)
    - dt: time step between data points (used to compute
        Kramers-Moyal coefficients)
        - this is the actual time elapsed between data points
            in the desired unit (e.g. minutes)
    - kernel_params: dictionary containing kernel parameters
        - bandwidth: float, bandwidth for kernel density estimation
        - kernel: str, type of kernel to use (e.g. 'gaussian')

    Outputs:
    - drift_km: numpy array, Kramers-Moyal drift estimate
        for each bin in feature space
    - diff_km: numpy array, Kramers-Moyal diffusion estimate
        for each bin in feature space
    """

    ndim = len(bins)
    powers = _get_km_powers(ndim)

    kmc = (
        _km_wrapper(
            traj_list,
            grads=d_traj_list,
            bins=bins,
            bw=kernel_params["bandwidth"],
            kernel=kernel_params["kernel"],
            powers=powers,
        )
        / dt
    )

    if ndim == 1:  # just need to take the first two rows
        drift_km = kmc[1]
        diff_km = kmc[2]
    else:  # if ndim > 1, need to make sure arrays are in the right shape
        # permuted axes (0, ndim, ndim-1, ..., 1)
        axes_permute = [0, *list(reversed(range(1, ndim + 1)))]
        #  swap last ndim axes to get correct shape:
        # n_powers x N[ndim] x N[ndim-1] x ... x N[1]
        kmc = np.transpose(kmc, axes_permute)
        # take drift terms, shape is N[1] x N[2] x ... x N[ndim] x ndim
        drift_km = kmc[1 : ndim + 1].T
        # take diffusion terms, shape is N[1] x N[2] x ... x N[ndim] x ndim
        diff_km = kmc[ndim + 1 :].T

    return drift_km, diff_km


def get_kramers_moyal(
    traj_list: list[np.ndarray],
    d_traj_list: list[np.ndarray],
    bins: list[np.ndarray],
    dt: float,
    kernel_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get estimation of Kramers-Moyal coefficients
    for drift and diffusion.

    Wrapper function for `_get_km_coeff`, which
    implements a kernel method for
    estimating Kramers-Moyal coefficients.
    These functions are defined in
    `km_kernels.py`.

    Inputs:
    - traj_list: list of numpy arrays, each array
        is a single trajectory in feature space
    - d_traj_list: list of numpy arrays, each array
        is the displacement vectors along that trajectory
    - bins: list of numpy arrays, each array contains
        the bin edges for a dimension
        (used for computing conditional averages)
    - dt: time step between data points
        (used to compute Kramers-Moyal coefficients)
    - kernel_params: dictionary of parameters for the kernel
        method for estimating Kramers-Moyal coefficients
        - bandwidth: bandwidth for the kernel method
        - kernel: kernel function to use for the kernel method

    Outputs:
    - drift_km: numpy array, drift estimates
        for each bin in feature space
    - diff_km: numpy array, diffusion estimates
        for each bin in feature space
    """
    if kernel_params is None:
        print("No kernel parameters provided, using default parameters: ")
        kernel_params = {"bandwidth": 0.1, "kernel": "gaussian"}
        print(
            f"bandwidth = {kernel_params['bandwidth']:.3f}," f"kernel = {kernel_params['kernel']}"
        )
    drift_km, diff_km = _get_km_coeff(traj_list, d_traj_list, bins, dt, kernel_params)
    return drift_km, diff_km
