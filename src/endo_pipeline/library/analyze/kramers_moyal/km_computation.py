import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.signal import convolve
from scipy.special import factorial

from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import histogramdd

logger = logging.getLogger(__name__)


def _check_and_adjust_km_inputs(
    trajectories: list[np.ndarray], displacements: list[np.ndarray], powers: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Check the inputs for the Kramers-Moyal coefficient computation and adjust them if necessary."""
    if len(trajectories) != len(displacements):
        raise ValueError("Must have displacements for each timeseries.")
    if len(displacements[0]) != len(trajectories[0]) - 1:
        raise ValueError(
            "Need to have displacements for each timepoint in timeseries except for last."
        )

    # check if timeseries is a list of 1D arrays, if so reshape to 2D arrays with one column
    if len(trajectories[0].shape) == 1:
        logger.warning(
            "Input ``timeseries`` is a list of 1D arrays. Reshaping to 2D arrays with one column."
        )
        for j, ts in enumerate(trajectories):
            trajectories[j] = ts.reshape(-1, 1)

    # get dimension of the timeseries
    ndim = trajectories[0].shape[1]

    # check if powers is a 1D array
    # if so, reshape it to a 2D array with one column
    if len(powers.shape) == 1:
        logger.warning("Input ``powers`` is a 1D array. Reshaping to 2D array with one column.")
        powers = powers.reshape(-1, 1)

    # add normalization factor to powers
    # if the first row is not all zeros
    if not (powers[0] == [0] * ndim).all():
        logger.warning(
            "First row of input ``powers`` is not all zeros. Adding zeros as first row for proper normalization."
        )
        powers = np.array([[0] * ndim, *powers])

    if powers.shape[1] != ndim:
        raise ValueError(
            "Array of powers must have the same number of columns as the dimension of the timeseries."
        )

    return trajectories, displacements, powers


def _get_km_powers(ndim: int) -> np.ndarray:
    """
    Generate the powers for the first two Kramers-Moyal coefficients for ndim
    dimensions.

    Note that for the second order Kramers-Moyal coefficients (diffusion), we
    only include the pure powers of each component (i.e., no interaction terms).

    For example, for 1D data, the powers are: [[0],  # normalization for kernel
    convolution (density)
     [1],  # drift [2]]  # diffusion

    For 2D data, the powers are: [[0,0],  # normalization for kernel convolution
    (density)
     [1,0],  # drift_1 [0,1],  # drift_2 [2,0],  # diffusion_11 [0,2]]  #
     diffusion_22
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


def _get_weighted_histogram_for_convolution(
    trajectories: list[np.ndarray],
    displacements: list[np.ndarray],
    bins: list[np.ndarray],
    powers: np.ndarray,
) -> np.ndarray:
    """
    Get the weighted histogram of the observations for convolution with the
    kernel function.

    This function computes the weighted histogram of the observations, where the
    weights are given by the products of the components of the displacement
    vectors raised to the corresponding powers specified in `powers`. The
    resulting weighted histogram is then used for convolution with the kernel
    function to estimate the Kramers-Moyal coefficients.

    For example, in 2D, suppose we input

        powers = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]],

    Then raising the concatenated displacements ``x(t+1)-x(t)` to the powers and
    taking the product across dimensions looks like:
        np.power(grads.T, powers[..., None]) = [[1, 1],
                                                [x_0(t+1)-x_0(t), 1] [1 ,
                                                x_1(t+1)-x_1(t)],
                                                [x_0(t+1)-x_0(t),
                                                (x_1(t+1)-x_1(t))],
                                                [(x_0(t+1)-x_0(t))^2, 1], [1,
                                                (x_1(t+1)-x_1(t))^2]]
        np.prod(..., axis=1) = [1,
                                x_0(t+1)-x_0(t), x_1(t+1)-x_1(t),
                                (x_0(t+1)-x_0(t))(x_1(t+1)-x_1(t)),
                                (x_0(t+1)-x_0(t))^2, (x_1(t+1)-x_1(t))^2]

    If there are L powers and M observations of the displacements, the result is
    an L x M array.
    """
    # Note that there is no corresponding displacement value for the last timepoint
    # of each trajectory, so we only take the timepoints up to the second to last one for the histogram.
    trajectories_concat = np.concatenate([ts[:-1] for ts in trajectories], axis=0)
    displacements_concat: np.ndarray = np.concatenate(displacements, axis=0)

    # Get weights for each displacement vector
    weights = np.prod(np.power(displacements_concat.T, powers[..., None]), axis=1)

    # Return the weighted histogram of the observations for convolution with the kernel function
    return histogramdd(trajectories_concat, bins=bins, weights=weights)


def _convolve_histogram_with_kernel(
    hist: np.ndarray,
    kernel_eval: np.ndarray,
    bins: list[np.ndarray],
    powers: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Compute the Kramers-Moyal coefficients by convolving the weighted histogram
    of observations with kernel.

    This function takes in the weighted histogram of the observations
    (displacements raised to powers) and the evaluated kernel on the extended
    histogram grid, and performs the convolution to compute the Kramers-Moyal
    coefficients. The coefficients are then normalized to ensure that the 0th
    order coefficient (probability density) integrates to 1, and the higher
    order coefficients are divided by the appropriate Taylor expansion
    coefficients and the 0th order coefficient.
    """

    # Convolve weighted histogram of kmc observations
    # mode "same" returns the convolution at the same size as
    # the input histogram, which is what we want
    kmc = convolve(hist, kernel_eval[None, ...], mode="same")

    # make sure that estimates are properly normalized
    norm_coeff = kmc[0].copy()
    for ii in range(len(bins)):
        bin_width = bins[ii][1] - bins[ii][0]
        norm_coeff = np.trapz(norm_coeff, dx=bin_width, axis=-1)
    kmc /= norm_coeff

    # Mask out bins where the probability density is smaller than the specified
    # tolerance
    mask = np.abs(kmc[0]) < tol
    kmc[0:, mask] = np.nan

    # get correct Taylor expansion coefficients (e.g., divide 2nd order powers
    # by 2!, etc.)

    # if we have higher order coefficients beyond the 0th order density
    # coefficient, then we need to divide by the appropriate Taylor expansion
    # coefficients and the 0th order coefficient to get the correct estimates of
    # the Kramers-Moyal coefficients.
    if powers.shape[0] > 1:
        taylors = np.prod(factorial(powers[1:]), axis=1)
        kmc[1:, ~mask] /= taylors[..., None] * kmc[0, ~mask]

    return kmc


def _reshape_outputs_to_drift_diffusion_coefficients(
    ndim: int, kmc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape the output of kernel convolution to get the drift and diffusion
    coefficients in the correct shape.
    """
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


def _compile_multivariate_product_kernel(
    kernels: list[Callable[[np.ndarray, float, float | None], np.ndarray]],
) -> Callable[[np.ndarray, list[float], list[float | None] | None], np.ndarray]:
    """
    Compile a multivariate kernel by taking the product of 1D kernels for each
    variable.

    This function allows for specifying different kernels and bandwidths for
    each variable/dimension, when performing multivariate kernel-based
    estimation.

    **Input kernels**

    The input `kernels` is a list of 1D scaled kernel functions, one for each
    variable/dimension. Each kernel function should take in an array of
    distances and a bandwidth, and return the scaled kernel values (see the
    `scaled_kernel` decorator for how to create such functions from basic kernel
    definitions).

    **Input to the resulting multivariate kernel function**

    The resulting multivariate kernel function will take in an array of
    differences along each dimension, where each row corresponds to the
    difference between a pair of points along each dimension, and a list of
    bandwidths for each variable/dimension. The function will evaluate the
    product of the kernel evaluations for each variable, using the specified
    kernels and bandwidths.

    Parameters
    ----------
    kernels
        List of 1D kernel functions, one for each variable/dimension.

    Returns
    -------
        A function that returns the product of the kernel evaluations for each
        variable.
    """

    def multivariate_kernel(
        x: np.ndarray, bw: list[float], period: list[float | None] | None = None
    ) -> np.ndarray:
        kernel_eval_list = []
        ndim = x.shape[-1]
        if ndim != len(bw):
            raise ValueError(
                f"Number of dimensions in input x ({ndim}) does not match number of bandwidths ({len(bw)})"
            )
        if period is not None and len(period) != ndim:
            raise ValueError(
                f"Number of dimensions in input x ({ndim}) does not match number of periods ({len(period)})"
            )
        for d in range(x.shape[-1]):
            kernel_eval = kernels[d](x[..., d], bw[d], period[d] if period is not None else None)
            kernel_eval_list.append(kernel_eval)

        kernel_product = np.prod(kernel_eval_list, axis=0)
        return kernel_product

    return multivariate_kernel


def _evaluate_single_multivariate_kernel(x: np.ndarray, kernel: KramersMoyalKernel) -> np.ndarray:
    """
    Evaluate a single multivariate kernel on the input array of differences
    along each dimension.
    """
    kernel_func = kernel.string_to_kernel()
    return kernel_func(x, kernel.bandwidth, kernel.period)


def _evaluate_multivariate_product_kernel(
    x: np.ndarray, kernel_list: list[KramersMoyalKernel]
) -> np.ndarray:
    """
    Evaluate a multivariate product kernel on the input array of differences
    along each dimension.
    """
    kernel_funcs = [kernel.string_to_kernel() for kernel in kernel_list]
    bandwidths = [kernel.bandwidth for kernel in kernel_list]
    periods = [kernel.period for kernel in kernel_list]
    kernel_func_prod = _compile_multivariate_product_kernel(kernel_funcs)

    # have to do some reshaping to properly evaluate the kernel at all points in the grid,
    # and then reshape back to the grid shape
    grid_shape = x.shape[:-1]
    ndim = x.shape[-1]
    return kernel_func_prod(x.reshape(-1, ndim), bandwidths, periods).reshape(grid_shape)


def get_cartesian_product(arrays: np.ndarray | list) -> np.ndarray:
    shapes = [len(arr) for arr in arrays]
    indices = np.indices(shapes, sparse=False)
    unstacked = [arrays[i][sub_indices] for i, sub_indices in enumerate(indices)]
    return np.stack(unstacked, axis=-1)


def get_kramers_moyal_coeffs(
    trajectories: list[np.ndarray],
    displacements: list[np.ndarray],
    bins: list[np.ndarray],
    dt: float,
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get estimates of first and second Kramers-Moyal coefficients from a list of
    trajectories.

    **Kernel specification**

    The input ``kernel`` can be a single kernel function that is applied to all
    dimensions, or a list of kernel functions for each dimension (in which case
    the product kernel is used).

    In general, the kernel is specified as a ``KramersMoyalKernel`` dataclass,
    which has attributes for the kernel name, bandwidth, and period (if
    applicable). If a list of kernels is provided, each kernel in the list
    should be a ``KramersMoyalKernel`` dataclass corresponding to each
    dimension.

    Parameters
    ----------
    trajectories
        List of invidual trajectories.
    displacements
        List of invidual displacements along the trajectories.
    bins
        List of monotonically increasing bin edges in each dimension.
    dt
        Time step between consecutive observations in the trajectories.
    kernel
        Kernel used to convolute with the Kramers-Moyal coefficients.

    Returns
    -------
    :
        First Kramers-Moyal coefficient (drift).
    :
        Second Kramers-Moyal coefficient (diffusion).
    """

    # get powers for first two Kramers-Moyal coefficients (drift and diffusion)
    # based on dimensionality of data
    ndim = len(bins)
    powers = _get_km_powers(ndim)

    # check inputs to avoid errors in the middle of the function
    trajectories, displacements, powers = _check_and_adjust_km_inputs(
        trajectories, displacements, powers
    )

    # get weighted histogram for convolution with kernel function
    hist = _get_weighted_histogram_for_convolution(trajectories, displacements, bins, powers)

    # Generate centered kernel on larger grid
    edges_extended = [(e[1] - e[0]) * np.arange(-e.size, e.size + 1) for e in bins]
    edges_cartesian_prod = get_cartesian_product(edges_extended)

    # Convert kernels into a callable and evaluate on the grid of points given by
    # the cartesian product of the extended edges.
    if isinstance(kernel, list):
        kernel_eval = _evaluate_multivariate_product_kernel(edges_cartesian_prod, kernel)
    else:
        kernel_eval = _evaluate_single_multivariate_kernel(edges_cartesian_prod, kernel)

    # Calculate coefficients by convolving histogram with kernel and divide by
    # dt to get the correct units (e.g. per minute)
    kmc = _convolve_histogram_with_kernel(hist, kernel_eval, bins, powers) / dt

    # reshape the output of kernel convolution to get just the drift and diffusion coefficients
    # and have them in the correct shape
    drift, diffusion = _reshape_outputs_to_drift_diffusion_coefficients(ndim, kmc)

    return drift, diffusion


def get_kernel_density_estimate_from_histogram(
    histogram: np.ndarray,
    bins: list[np.ndarray],
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
) -> np.ndarray:
    """
    Get kernel density estimate of a given histogram.

    Parameters
    ----------
    histogram
        n-dimensional histogram of the observations, where the bins are defined
        by the input `bins`.
    bins
        List of monotonically increasing bin edges in each dimension.
    kernel
        Kernel used to convolute with the histogram of observations to get the
        kernel density estimate.

    Returns
    -------
    :
        Kernel density estimate of the n-dimensional joint probability density
        of the observations.
    """
    # get powers for first two Kramers-Moyal coefficients (drift and diffusion)
    # based on dimensionality of data, power for 0th order coefficient (density) is all zeros
    ndim = len(bins)
    histogram = np.atleast_2d(histogram)  # make sure histogram is at least 2D for convolution
    powers = np.zeros((1, ndim), dtype=int)

    # Generate centered kernel on larger grid
    edges_extended = [(e[1] - e[0]) * np.arange(-e.size, e.size + 1) for e in bins]
    edges_cartesian_prod = get_cartesian_product(edges_extended)

    # Convert kernels into a callable and evaluate on the grid of points given by
    # the cartesian product of the extended edges.
    if isinstance(kernel, list):
        kernel_eval = _evaluate_multivariate_product_kernel(edges_cartesian_prod, kernel)
    else:
        kernel_eval = _evaluate_single_multivariate_kernel(edges_cartesian_prod, kernel)

    # Calculate KDE by convolving histogram with kernel
    prob_kde = _convolve_histogram_with_kernel(histogram, kernel_eval, bins, powers)[0]

    return prob_kde


def get_kernel_density_estimate_from_trajectories(
    trajectories: list[np.ndarray],
    bins: list[np.ndarray],
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
) -> np.ndarray:
    """
    Get kernel density estimate of the probability density of the observations.

    This function is similar to `get_kramers_moyal_coeffs`, but only returns the
    0th order Kramers-Moyal coefficient, which corresponds to the probability
    density of the observations. This can be useful for visualizing the density
    of the data in state space, and for masking out regions with low density
    when visualizing the drift and diffusion estimates.

    Parameters
    ----------
    trajectories
        List of invidual n-dimensional trajectories (observations).
    bins
        List of monotonically increasing bin edges in each dimension.
    kernel
        Kernel used to convolute with the histogram of observations to get the
        kernel density estimate.

    Returns
    -------
    :
        Kernel density estimate of the n-dimensional joint probability density
        of the observations.
    """
    # get powers for first two Kramers-Moyal coefficients (drift and diffusion)
    # based on dimensionality of data
    ndim = len(bins)
    powers = np.zeros(
        (1, ndim), dtype=int
    )  # power for 0th order coefficient (density) is all zeros

    # check inputs to avoid errors in the middle of the function
    displacements = [
        np.zeros_like(ts[:-1]) for ts in trajectories
    ]  # dummy displacements since we only care about density of observations
    trajectories, displacements, powers = _check_and_adjust_km_inputs(
        trajectories, displacements, powers
    )

    # get weighted histogram for convolution with kernel function
    hist = _get_weighted_histogram_for_convolution(trajectories, displacements, bins, powers)

    # Call method that convolves histogram with kernel to get Kramers-Moyal
    # coefficients, and take the 0th order coefficient which corresponds to the
    # probability density of the observations.
    prob_kde = get_kernel_density_estimate_from_histogram(hist, bins, kernel)

    return prob_kde


def _get_bin_indices(points: np.ndarray, bins: list[np.ndarray]) -> np.ndarray | None:
    """
    Map an array of points to their bin indices using the provided bin edges.

    Parameters
    ----------
    points
        Array of shape ``(n_points, ndim)`` containing the coordinates of the
        points to look up.
    bins
        List of monotonically increasing bin-edge arrays for each dimension,
        matching the convention used throughout this module (i.e., N bins are
        defined by N+1 edges).

    Returns
    -------
    :
        Integer array of shape ``(n_valid_points, ndim)`` containing the
        0-indexed bin index for each in-range point, or ``None`` if no points
        fall within the bin range.
    """
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    ndim = points.shape[1]
    n_points = points.shape[0]
    bin_indices = np.zeros((n_points, ndim), dtype=int)
    in_range = np.ones(n_points, dtype=bool)

    for d in range(ndim):
        edges = bins[d]
        coords = points[:, d]

        # np.digitize returns 1-indexed positions; values below edges[0] get
        # index 0, values >= edges[-1] get index len(edges).
        idx = np.digitize(coords, edges) - 1  # convert to 0-indexed

        # Valid bin indices are 0 .. len(edges)-2 (N bins for N+1 edges)
        out_of_range = (idx < 0) | (idx >= len(edges) - 1)
        in_range &= ~out_of_range
        bin_indices[:, d] = np.clip(idx, 0, len(edges) - 2)

    if in_range.sum() == 0:
        return None

    return bin_indices[in_range]


def _score_fold(
    k_idx: int,
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
    fold: np.ndarray,
    shuffled_indices: np.ndarray,
    trajectories: list[np.ndarray],
    bins: list[np.ndarray],
    displacements: list[np.ndarray] | None,
    dt: float | None,
) -> tuple[int, float]:
    """
    Score a single (kernel candidate, fold) pair.

    This is a module-level function so it can be pickled by
    :class:`concurrent.futures.ProcessPoolExecutor`.

    Returns
    -------
    k_idx :
        Index of the kernel candidate (passed through for result aggregation).
    log_lik :
        Mean log-likelihood of held-out observations under the model fitted on
        the training split, or ``-inf`` if the fold could not be scored.
    """
    km_mode = displacements is not None
    train_indices = np.setdiff1d(shuffled_indices, fold)
    if len(train_indices) == 0:
        return k_idx, -np.inf

    train_trajectories = [trajectories[i] for i in train_indices]

    if km_mode:
        assert displacements is not None and dt is not None  # type narrowing
        train_displacements = [displacements[i] for i in train_indices]

        drift, diffusion = get_kramers_moyal_coeffs(
            train_trajectories, train_displacements, bins, dt, kernel
        )

        held_out_positions = np.concatenate(
            [
                (
                    trajectories[i][:-1]
                    if trajectories[i].ndim > 1
                    else trajectories[i][:-1].reshape(-1, 1)
                )
                for i in fold
            ],
            axis=0,
        )
        held_out_displacements = np.concatenate(
            [
                (displacements[i] if displacements[i].ndim > 1 else displacements[i].reshape(-1, 1))
                for i in fold
            ],
            axis=0,
        )

        valid_bin_indices = _get_bin_indices(held_out_positions, bins)
        if valid_bin_indices is None:
            return k_idx, -np.inf

        ndim = len(bins)
        idx_tuple = tuple(valid_bin_indices[:, d] for d in range(ndim))

        if ndim == 1:
            f_vals = drift[idx_tuple[0], np.newaxis]  # (n_valid, 1)
            D_vals = diffusion[idx_tuple[0], np.newaxis]
        else:
            f_vals = drift[idx_tuple]  # (n_valid, ndim)
            D_vals = diffusion[idx_tuple]

        dx_vals = held_out_displacements[
            np.all(
                (held_out_positions >= np.array([b[0] for b in bins]))
                & (held_out_positions < np.array([b[-1] for b in bins])),
                axis=1,
            )
        ]

        valid_mask = ~(np.isnan(f_vals).any(axis=1) | np.isnan(D_vals).any(axis=1))
        valid_mask &= (D_vals > 0).all(axis=1)

        if not valid_mask.any():
            return k_idx, -np.inf

        f_v = f_vals[valid_mask]
        D_v = D_vals[valid_mask]
        dx_v = dx_vals[valid_mask]

        log_liks_per_dim = -0.5 * np.log(4.0 * np.pi * D_v * dt) - (dx_v - f_v * dt) ** 2 / (
            4.0 * D_v * dt
        )
        return k_idx, float(np.mean(log_liks_per_dim.sum(axis=1)))

    else:
        kde = get_kernel_density_estimate_from_trajectories(train_trajectories, bins, kernel)

        held_out_points = np.concatenate(
            [
                (trajectories[i] if trajectories[i].ndim > 1 else trajectories[i].reshape(-1, 1))
                for i in fold
            ],
            axis=0,
        )

        valid_bin_indices = _get_bin_indices(held_out_points, bins)
        if valid_bin_indices is None:
            return k_idx, -np.inf

        kde_vals = kde[tuple(valid_bin_indices[:, d] for d in range(valid_bin_indices.shape[1]))]

        positive = kde_vals > 0
        if not positive.any():
            return k_idx, -np.inf
        return k_idx, float(np.mean(np.log(kde_vals[positive])))


def select_bandwidth_cross_validation(
    trajectories: list[np.ndarray],
    bins: list[np.ndarray],
    kernel_candidates: list[KramersMoyalKernel | list[KramersMoyalKernel]],
    n_splits: int = 5,
    random_state: int | None = None,
    displacements: list[np.ndarray] | None = None,
    dt: float | None = None,
    n_jobs: int = 1,
) -> tuple[KramersMoyalKernel | list[KramersMoyalKernel], np.ndarray]:
    r"""
    Select the kernel specification by k-fold cross-validation.

    Trajectories are divided into ``n_splits`` folds at the trajectory level.
    For each candidate kernel specification, a model is fitted on the training
    folds and scored on the held-out fold. The candidate that maximises the
    cross-validated log-likelihood is returned.

    **Kernel specification**

    Each element of ``kernel_candidates`` follows the same convention as the
    ``kernel`` argument to :func:`get_kramers_moyal_coeffs` and
    :func:`get_kernel_density_estimate_from_trajectories`:

    * A single :class:`KramersMoyalKernel` — the same kernel (name, bandwidth,
      and period) is applied isotropically across all dimensions.
    * A list of :class:`KramersMoyalKernel` — one kernel per dimension,
      allowing different kernel types, bandwidths, or periods for each
      coordinate (e.g., a ``"periodic"`` kernel for polar angle and a
      ``"gaussian"`` kernel for radial distance).

    **Scoring mode**

    The scoring mode is determined by whether ``displacements`` and ``dt`` are
    provided:

    * **KDE mode** (default — ``displacements`` and ``dt`` omitted): fits a
      kernel density estimate on training trajectories and evaluates the mean
      log-likelihood of held-out observations under that density.

    * **Kramers-Moyal mode** (``displacements`` and ``dt`` supplied): fits the
      Kramers-Moyal drift and diffusion coefficients on training data and scores
      each held-out displacement :math:`dx_t` at position :math:`x_t` using the
      Gaussian Langevin log-likelihood

      .. math::

          \\log p(dx_t \\mid x_t) = \\sum_d \\left[
              -\\tfrac{1}{2}\\log(4\\pi D_d(x_t)\\,dt)
              - \\frac{(dx_{t,d} - f_d(x_t)\\,dt)^2}{4\\,D_d(x_t)\\,dt}
          \\right]

      where :math:`f_d` and :math:`D_d` are the estimated drift and diffusion
      along dimension :math:`d`. Observations falling in bins where either
      coefficient is NaN (low-density regions) are excluded from the score.

    Parameters
    ----------
    trajectories
        List of individual trajectories.  Each trajectory is an array of shape
        ``(T,)`` (1-D) or ``(T, ndim)`` (multi-D).
    bins
        List of monotonically increasing bin-edge arrays in each dimension,
        matching the convention used by
        :func:`get_kernel_density_estimate_from_trajectories` and
        :func:`get_kramers_moyal_coeffs`.
    kernel_candidates
        List of candidate kernel specifications to evaluate.  Each element is
        either a single :class:`KramersMoyalKernel` (isotropic) or a list of
        :class:`KramersMoyalKernel` objects with one entry per dimension
        (anisotropic / mixed-kernel).
    n_splits
        Number of cross-validation folds (capped at the number of
        trajectories). Defaults to ``5``.
    random_state
        Seed for the random number generator used to shuffle trajectories
        before splitting, enabling reproducible results.
    displacements
        List of displacement arrays, one per trajectory, matching the
        convention used by :func:`get_kramers_moyal_coeffs`.  Providing this
        argument (together with ``dt``) switches to **Kramers-Moyal mode**.
    dt
        Time step between consecutive observations.  Required when
        ``displacements`` is provided.
    n_jobs
        Number of worker processes to use for parallelising over
        ``(kernel_candidate, fold)`` pairs.

        * ``1`` (default) — run sequentially, no extra processes.
        * ``-1`` — use all available CPUs (``os.cpu_count()``).
        * Any positive integer — use that many worker processes.

        Each worker receives a copy of the full trajectory data via pickle,
        so very large datasets may see diminishing returns.  Profile before
        setting ``n_jobs`` to a large value.

    Returns
    -------
    best_kernel :
        The element of ``kernel_candidates`` with the highest cross-validated
        log-likelihood.
    cv_scores :
        Array of mean cross-validated log-likelihoods, one entry per candidate
        in ``kernel_candidates``, in the same order.

    Raises
    ------
    ValueError
        If fewer than 2 trajectories are provided, or if exactly one of
        ``displacements`` / ``dt`` is supplied without the other.

    Examples
    --------
    KDE mode — isotropic Gaussian kernel, varying bandwidth:

    >>> import numpy as np
    >>> from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    ...     select_bandwidth_cross_validation,
    ... )
    >>> from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    >>> rng = np.random.default_rng(0)
    >>> trajectories = [rng.standard_normal((50, 1)) for _ in range(20)]
    >>> bins = [np.linspace(-4, 4, 41)]
    >>> candidates = [KramersMoyalKernel("gaussian", bw) for bw in [0.05, 0.1, 0.2, 0.4]]
    >>> best_kernel, scores = select_bandwidth_cross_validation(
    ...     trajectories, bins, candidates, n_splits=5
    ... )

    KM mode — per-dimension kernels (periodic angle + Gaussian radius):

    >>> from numpy import pi
    >>> bins_2d = [np.linspace(0, pi, 51), np.linspace(0, 3.5, 36)]
    >>> displacements = [np.diff(traj, axis=0) for traj in trajectories]
    >>> candidates_2d = [
    ...     [KramersMoyalKernel("periodic", bw, period=pi), KramersMoyalKernel("gaussian", bw)]
    ...     for bw in [0.05, 0.1, 0.2, 0.4]
    ... ]
    >>> best_kernel, scores = select_bandwidth_cross_validation(
    ...     trajectories, bins_2d, candidates_2d,
    ...     displacements=displacements, dt=1.0, n_splits=5,
    ... )
    """
    # --- input validation ---
    if (displacements is None) != (dt is None):
        raise ValueError(
            "Both ``displacements`` and ``dt`` must be provided together to use "
            "Kramers-Moyal mode, or neither should be provided for KDE mode."
        )
    km_mode = displacements is not None

    rng = np.random.default_rng(random_state)
    n_trajectories = len(trajectories)
    if n_trajectories < 2:
        raise ValueError("At least 2 trajectories are required for cross-validation.")

    n_splits = min(n_splits, n_trajectories)

    # Shuffle trajectory indices and split into folds
    shuffled_indices = rng.permutation(n_trajectories)
    folds: list[np.ndarray] = np.array_split(shuffled_indices, n_splits)

    # Common keyword arguments forwarded to every _score_fold call
    _fold_kwargs = {
        "shuffled_indices": shuffled_indices,
        "trajectories": trajectories,
        "bins": bins,
        "displacements": displacements,
        "dt": dt,
    }

    # Build the flat list of (k_idx, kernel, fold) jobs
    jobs = [
        (k_idx, kernel, fold) for k_idx, kernel in enumerate(kernel_candidates) for fold in folds
    ]

    # Accumulate per-fold log-likelihoods keyed by k_idx
    fold_log_liks_by_k: dict[int, list[float]] = {k: [] for k in range(len(kernel_candidates))}

    max_workers = None if n_jobs == -1 else (n_jobs if n_jobs > 1 else None)

    if n_jobs == 1:
        # --- sequential ---
        for k_idx, kernel, fold in jobs:
            _, log_lik = _score_fold(k_idx, kernel, fold, **_fold_kwargs)
            if np.isfinite(log_lik):
                fold_log_liks_by_k[k_idx].append(log_lik)
            else:
                logger.warning(
                    "Fold could not be scored for kernel candidate %d; fold skipped.", k_idx
                )
    else:
        # --- parallel ---
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_k = {
                executor.submit(_score_fold, k_idx, kernel, fold, **_fold_kwargs): k_idx
                for k_idx, kernel, fold in jobs
            }
            for future in as_completed(future_to_k):
                k_idx_result, log_lik = future.result()
                if np.isfinite(log_lik):
                    fold_log_liks_by_k[k_idx_result].append(log_lik)
                else:
                    logger.warning(
                        "Fold could not be scored for kernel candidate %d; fold skipped.",
                        k_idx_result,
                    )

    cv_scores = np.array(
        [
            np.mean(fold_log_liks_by_k[k]) if fold_log_liks_by_k[k] else -np.inf
            for k in range(len(kernel_candidates))
        ]
    )

    best_idx = np.argmax(cv_scores)
    best_kernel = kernel_candidates[best_idx]
    mode_label = "KM" if km_mode else "KDE"
    logger.info(
        "[%s] Cross-validation selected kernel candidate %d (CV log-likelihood: %.4g).",
        mode_label,
        best_idx,
        cv_scores[best_idx],
    )
    return best_kernel, cv_scores
