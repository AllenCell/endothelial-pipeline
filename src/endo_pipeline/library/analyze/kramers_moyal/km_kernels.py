import inspect
import logging
from collections.abc import Callable
from functools import wraps

import numpy as np
from scipy.special import gamma

AVAILABLE_KERNEL_FUNCTIONS = ["epanechnikov", "gaussian"]

logger = logging.getLogger(__name__)


def string_to_kernel(kernel: str) -> Callable:
    """
    Convert a kernel name to the corresponding callable (scaled) kernel function.
    """
    # get dictionary of all callable kernel functions in this module
    import sys

    kernel_dict = {
        name: func
        for name, func in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name in AVAILABLE_KERNEL_FUNCTIONS
    }
    if kernel in kernel_dict:
        return kernel_dict[kernel]
    else:
        raise ValueError(
            f"Kernel '{kernel}' not recognized. " f" Available kernels: {list(kernel_dict.keys())}"
        )


def _volume_unit_ball(dims: int) -> float:
    # volume of a unit ball in dimension dims
    return np.pi ** (dims / 2.0) / gamma(dims / 2.0 + 1.0)


def _get_input_dims_and_distances(x: np.ndarray) -> tuple[int, np.ndarray]:
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

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
