import numpy as np

import cellsmap.analyses.utils.numerics.kramersmoyal.kmc as km


def get_km_powers(ndim: int) -> np.ndarray:
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


def get_km_kernel(
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
    powers = get_km_powers(ndim)

    kmc = (
        km.km(
            traj_list,
            grads=d_traj_list,
            bins=bins,
            bw=kernel_params["bandwidth"],
            kernel=kernel_params["kernel"],
            powers=powers,
            multi_traj=True,
        )
        / dt
    )

    if ndim == 1:  # just need to take the first two rows
        drift_km = kmc[1]
        diff_km = kmc[2]
    else:  # if ndim > 1, need to make sure arrays are in the right shape
        # permuted axes (0, ndim, ndim-1, ..., 1)
        axes_permute = [0] + list(
            reversed(range(1, ndim + 1))
        )
        #  swap last ndim axes to get correct shape:
        # n_powers x N[ndim] x N[ndim-1] x ... x N[1]
        kmc = np.transpose(kmc, axes_permute)
        # take drift terms, shape is N[1] x N[2] x ... x N[ndim] x ndim
        drift_km = kmc[1 : ndim + 1].T
        # take diffusion terms, shape is N[1] x N[2] x ... x N[ndim] x ndim
        diff_km = kmc[ndim + 1 :].T

    return drift_km, diff_km
