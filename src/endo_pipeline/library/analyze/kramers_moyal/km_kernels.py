import logging
from collections.abc import Callable
from functools import wraps

import numpy as np
from scipy.special import gamma

logger = logging.getLogger(__name__)


def _volume_unit_ball(dims: int) -> float:
    """Calculate the volume of a unit ball in a given number of dimensions."""
    # volume of a unit ball in dimension dims
    return np.pi ** (dims / 2.0) / gamma(dims / 2.0 + 1.0)


def _get_input_dims_and_distances(x: np.ndarray) -> tuple[int, np.ndarray]:
    """Get the number of dimensions and the Euclidean norm of the input array."""
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    # x is an array of shape (n_points, n_dims), where each row corresponds
    # to the difference between a pair of points along each dimension
    dims = x.shape[-1]

    # Euclidean norm of the array of vector differences
    euc_norm = np.sqrt((x * x).sum(axis=-1))

    return dims, euc_norm


def scaled_kernel(kernel_func: Callable) -> Callable:
    """
    Transform a pre-defined kernel function into a scaled kernel function
    that can be used for kernel density estimation.

    **Original kernel function**

    The original kernel function ``kernel_func`` should take in an array of distances and the
    number of dimensions, and return the kernel values. Specifically, the array of distances
    is an m x n array, where m is the number of pairs of points and n is the number of dimensions.
    Then row i of the array corresponds to the difference between the i-th pair of points along each dimension.

    **Kernel evaluation and scaling**

    Using this decorator, the resulting scaled kernel function will take in an said
    array of distances and a bandwidth, compute the norm of the distances (i.e., turn x-y to ||x-y||),
    and return the scaled kernel values. The scaling is done by dividing the distances by the bandwidth,
    and then normalizing by the bandwidth raised to the power of the number of dimensions.

    The value is also divided by the volume of the unit ball in that number of dimensions, so that
    resulting kernel function can be used for kernel density estimation in any number of dimensions.
    """

    @wraps(kernel_func)  # just for naming
    def decorated(x: np.ndarray, bw: float) -> np.ndarray:
        dims, dist = _get_input_dims_and_distances(x)
        return kernel_func(dist / bw) / (bw**dims) / _volume_unit_ball(dims)

    return decorated


@scaled_kernel
def epanechnikov(x: np.ndarray) -> np.ndarray:
    """Define the Epanechnikov kernel."""
    x2 = x**2
    mask = x2 < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = 1.0 - x2[mask]
    return kernel


@scaled_kernel
def gaussian(x: np.ndarray) -> np.ndarray:
    """Define the Gaussian kernel."""
    kernel = np.exp(-(x**2) / 2.0) / np.sqrt(2 * np.pi)
    return kernel


AVAILABLE_KERNEL_FUNCTIONS = {"epanechnikov": epanechnikov, "gaussian": gaussian}


def string_to_kernel(kernel: str) -> Callable:
    """
    Convert a kernel name to the corresponding callable (scaled) kernel function.
    """
    # check if kernel is in the available implemented kernels,
    # and return the corresponding function
    if kernel in AVAILABLE_KERNEL_FUNCTIONS.keys():
        return AVAILABLE_KERNEL_FUNCTIONS[kernel]
    else:
        raise ValueError(
            f"Kernel '{kernel}' not recognized. "
            f" Available kernels: {list(AVAILABLE_KERNEL_FUNCTIONS.keys())}"
        )


def compile_multivariate_product_kernel(
    kernels: list[Callable[[np.ndarray, float], np.ndarray]],
    bandwidths: list[float],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Compile a multivariate kernel by taking the product of 1D kernels for each variable.

    This function allows for specifying different kernels and bandwidths for each variable/dimension,
    when performing multivariate kernel-based estimation.

    **Input kernels and bandwidths**

    The input `kernels` is a list of 1D scaled kernel functions, one for each variable/dimension.
    Each kernel function should take in an array of distances and a bandwidth, and return
    the scaled kernel values (see the `scaled_kernel` decorator for how to create such functions
    from basic kernel definitions).

    The input `bandwidths` is a list of bandwidths for each variable/dimension, which will be passed
    to the corresponding kernel function for that variable.

    **Input to the resulting multivariate kernel function**

    The resulting multivariate kernel function will take in an array of differences along each dimension,
    where each row corresponds to the difference between a pair of points along each dimension.
    The function will evaluate the product of the kernel evaluations for each variable, using the
    specified kernels and bandwidths.

    Parameters
    ----------
    kernels
        List of 1D kernel functions, one for each variable/dimension.
    bandwidths
        List of bandwidths for each variable/dimension.

    Returns
    -------
        A function that returns the product of the kernel evaluations for each variable.
    """

    def multivariate_kernel(x: np.ndarray) -> np.ndarray:
        kernel_eval_list = []
        for d in range(x.shape[-1]):
            kernel_eval = kernels[d](x[..., d], bandwidths[d])
            kernel_eval_list.append(kernel_eval)

        kernel_product = np.prod(kernel_eval_list, axis=0)
        return kernel_product

    return multivariate_kernel
