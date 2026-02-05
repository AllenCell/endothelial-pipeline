import inspect
from collections.abc import Callable

import numpy as np
from scipy.signal import convolve
from scipy.special import factorial

from endo_pipeline.library.analyze.kramers_moyal import km_kernels
from endo_pipeline.library.analyze.numerics import histogramdd


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


def _check_and_adjust_km_inputs(
    trajectories: list[np.ndarray], displacements: list[np.ndarray], powers: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    if len(trajectories) != len(displacements):
        raise ValueError("Must have displacements for each timeseries.")
    if len(displacements[0]) != len(trajectories[0]) - 1:
        raise ValueError(
            "Need to have displacements for each timepoint in timeseries except for last."
        )

    # check if timeseries is a list of 1D arrays, if so reshape to 2D arrays with one column
    if len(trajectories[0].shape) == 1:
        for j, ts in enumerate(trajectories):
            trajectories[j] = ts.reshape(-1, 1)

    # get dimension of the timeseries
    ndim = trajectories[0].shape[1]

    # check if powers is a 1D array
    # if so, reshape it to a 2D array with one column
    if len(powers.shape) == 1:
        powers = powers.reshape(-1, 1)

    # add normalization factor to powers
    # if the first row is not all zeros
    if not (powers[0] == [0] * ndim).all():
        powers = np.array([[0] * ndim, *powers])

    if powers.shape[1] != ndim:
        raise ValueError(
            "Array of powers must have the same number of columns as the dimension of the timeseries."
        )

    return trajectories, displacements, powers


def _km_wrapper(
    trajectories: list[np.ndarray],
    displacements: list[np.ndarray],
    bins: list[np.ndarray],
    powers: np.ndarray,
    kernel: str,
    kernel_bw: float,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Estimates the Kramers─Moyal coefficients from a timeseries using a kernel
    estimator method. `km` can calculate the Kramers─Moyal coefficients for a
    timeseries of any dimension, up to any desired power.

    **Binning**

    The input array ``bins`` specifies the bin edges in each dimension of the timeseries.
    The Kramers─Moyal coefficients are estimated at the center of each bin, and the
    output `kmc` is given in the shape of the number of bins in each dimension.

    **Power of the Kramers─Moyal coefficients**

    The input array ``powers`` specifies the exponents of the components
    of the displacement vectors in the Kramers─Moyal coefficients.

    The powers are given in the form of a 2-D array, where each row corresponds
    to a power, and each column corresponds to a component of the displacement vector
    that is raised to that power. The first row is always zero, to account for
    the normalization of the coefficients.

    For example, to compute each component of the drift coefficient in 2D, the powers
    are `[[0, 0], [1, 0], [0, 1]]`, where the first row is the normalization.

    The powers can be computed by calling ``_get_km_powers(ndim)``, where ``ndim`` is
    the dimension of the timeseries. The powers are then automatically generated,
    but only up to second order (drift and diffusion coefficients).

    The order that they appear dictates the order of the corresponding coefficient
    in the output ``kmc``. To extract the selected orders of the kmc, use ``kmc[i,...]``,
    with ``i`` being the order according to powers.

    **Kernel**

    The input string ``kernel`` specifies the kernel function that is used to compute
    the Kramers─Moyal coefficients as a convolution of the kernel with the weighted
    histogram of the observations. This input has to be a string that corresponds to a
    kernel function defined in the `km_kernels` module.

    **Low probability density**

    The input ``tol`` specifies the tolerance for small values of the probability density,
    which is the 0th order Kramers─Moyal coefficient. In bins where the probability density
    is smaller than this tolerance, the Kramers─Moyal coefficients are set to NaN.

    **References**

    [Lamouroux2009] D. Lamouroux and K. Lehnertz, "Kernel-based regression of
    drift and diffusion coefficients of stochastic processes." Physics Letters A
    373(39), 3507─3512, 2009. DOI: 10.1016/j.physleta.2009.07.073

    [Gorjão2019] L. R. Gorjão and F. Meirinhos, "kramersmoyal: Kramers-Moyal
    coefficients for stochastic processes." Journal of Open Source Software 4(44),
    1693, 2019. DOI: 10.21105/joss.01693

    Parameters
    ----------
    trajectories
        List of invidual trajectories.

    displacements
        List of invidual displacements along the trajectories.

    bins: list of np.ndarrays
        List of monotonically increasing bin edges in each dimension.

    powers
        Powers for the operation of calculating the Kramers─Moyal coefficients.

    kernel
        Kernel used to convolute with the Kramers-Moyal coefficients.

    bw
        Desired bandwidth of the kernel.

    tol
        Tolerance for small values of the probability density (0th order Kramers─Moyal coefficient).
    """
    # check inputs to avoid errors in the middle of the function
    trajectories, displacements, powers = _check_and_adjust_km_inputs(
        trajectories, displacements, powers
    )

    # convert specified kernel to callable
    kernel_func = _string_to_kernel(kernel)

    # Get trajectories and corresponding displacements concatenated across all trajectories.
    # Note that the last timepoint of each trajectory is not included,
    # as there is no corresponding displacement value for it.
    trajectories_concat = np.concatenate([ts[:-1] for ts in trajectories], axis=0)
    displacements_concat: np.ndarray = np.concatenate(displacements, axis=0)

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
    weights = np.prod(np.power(displacements_concat.T, powers[..., None]), axis=1)

    ##### Get weighted histogram for convolution

    # If there are L powers, the result in an L x N[0] x N[1] x ... x N[D-1] array
    # where N[i] is the number of bins in dimension i.
    hist = histogramdd(trajectories_concat, bins=bins, weights=weights)

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
    edges_k = [(e[1] - e[0]) * np.arange(-e.size, e.size + 1) for e in bins]
    kernel_eval = kernel_func(get_cartesian_product(edges_k), bw=kernel_bw)

    ##### KMC computation: convolve the histogram with the kernel

    # Convolve weighted histogram of kmc observations (displacements ^ powers)
    # with augmented periodic kernel and trim it back to the original size.
    # Note that the first entry of the output is the normalization factor
    # of the kernel estimator, which is the same for all dimensions.
    # The normalization factor is the KDE of the empirical density function.
    kmc = convolve(hist, kernel_eval[None, ...], mode="same")

    # Normalise with correct factorial coefficients * histogram
    mask = np.abs(kmc[0]) < tol  # where probability density is small... (i.e., little to no data)
    kmc[0:, mask] = np.nan  # ...set kmc coeffs to nan

    # get correct Taylor expansion coefficients (e.g., divide 2nd order powers by 2!)
    taylors = np.prod(factorial(powers[1:]), axis=1)
    kmc[1:, ~mask] /= (
        taylors[..., None] * kmc[0, ~mask]
    )  # divide by Taylor coeff * 0th order coeffs (probability density)

    return kmc


def get_cartesian_product(arrays: np.ndarray | list) -> np.ndarray:
    shapes = [len(arr) for arr in arrays]
    indices = np.indices(shapes, sparse=False)
    unstacked = [arrays[i][sub_indices] for i, sub_indices in enumerate(indices)]
    return np.stack(unstacked, axis=-1)


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


def get_kramers_moyal_coeffs(
    trajectories: list[np.ndarray],
    displacements: list[np.ndarray],
    bins: list[np.ndarray],
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

    # get powers for first two Kramers-Moyal coefficients (drift and diffusion)
    # based on dimensionality of data
    ndim = len(bins)
    powers = _get_km_powers(ndim)

    # compute Kramers-Moyal coefficients using kernel estimator method,
    # and divide by dt to get the correct units (e.g. per minute)
    kmc = (
        _km_wrapper(
            trajectories,
            displacements,
            bins=bins,
            kernel_bw=kernel_params["bandwidth"],
            kernel=kernel_params["kernel"],
            powers=powers,
        )
        / dt
    )

    # reshape the output to get drift and diffusion coefficients
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
