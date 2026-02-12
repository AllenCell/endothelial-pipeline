import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import numpy as np
from scipy.special import gamma

logger = logging.getLogger(__name__)


def _volume_unit_ball(dims: int) -> float:
    """Calculate the volume of a unit ball in a given number of dimensions."""
    # volume of a unit ball in dimension dims
    return np.pi ** (dims / 2.0) / gamma(dims / 2.0 + 1.0)


def _get_input_dims_and_distances(
    x: np.ndarray, period: float | None = None
) -> tuple[int, np.ndarray]:
    """Get the number of dimensions and the Euclidean norm of the input array."""
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    # x is an array of shape (n_points, n_dims), where each row corresponds
    # to the difference between a pair of points along each dimension
    dims = x.shape[-1]

    # Euclidean norm of the array of vector differences
    distances = np.sqrt((x * x).sum(axis=-1))

    # if period is not None, use the sine squared distance for periodic data
    #  (for implementation of the the exp sine squared kernel)
    if period is not None:
        # sine squared of distance for periodic data
        # (i.e., the exp sine squared kernel)
        distances = np.sin(np.pi * distances / period) ** 2

    return dims, distances


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
    def decorated(x: np.ndarray, bw: float, period: float | None = None) -> np.ndarray:
        dims, dist = _get_input_dims_and_distances(x, period)
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


@scaled_kernel
def periodic(x: np.ndarray) -> np.ndarray:
    """
    Define the periodic (exponential sine squared) kernel.

    Differs from the Gaussian kernel in that the exponential factor is
    -2 times the sine squared of the distance, rather than -0.5 times
    the squared distance.

    Input x should be the sin(pi * distance / period), where ``distance``
    is the Euclidean norm of the difference between points.
    """
    kernel = np.exp(-2 * (x**2))
    return kernel


AVAILABLE_KERNEL_FUNCTIONS = {
    "epanechnikov": epanechnikov,
    "gaussian": gaussian,
    "periodic": periodic,
}


@dataclass(frozen=True)
class KramersMoyalKernel:
    """Structure for kernels used to calculate Kramers-Moyal coefficients."""

    name: Literal["epanechnikov", "gaussian", "periodic"]
    """Name of the kernel."""

    bandwidth: float
    """Kernel bandwidth."""

    period: float | None = None
    """Kernel period (only required for periodic kernel)."""

    def __post_init__(self) -> None:
        """Validate kernel name, bandwidth, and period."""
        if self.name not in AVAILABLE_KERNEL_FUNCTIONS.keys():
            raise ValueError(
                f"Kernel '{self.name}' not recognized. "
                f" Available kernels: {list(AVAILABLE_KERNEL_FUNCTIONS.keys())}"
            )
        if self.name == "periodic" and self.period is None:
            raise ValueError("Period must be specified for periodic kernel.")
        if self.name != "periodic" and self.period is not None:
            raise ValueError("Period should not be specified for non-periodic kernels.")
        if self.period is not None and self.period <= 0:
            raise ValueError(f"Period must be positive, got {self.period}")
        if self.bandwidth <= 0:
            raise ValueError(f"Bandwidth must be positive, got {self.bandwidth}")

    def string_to_kernel(self) -> Callable[[np.ndarray, float, float | None], np.ndarray]:
        """Convert the kernel name to the corresponding callable (scaled) kernel function."""
        return AVAILABLE_KERNEL_FUNCTIONS[self.name]


def compile_multivariate_product_kernel(
    kernels: list[Callable[[np.ndarray, float, float | None], np.ndarray]],
) -> Callable[[np.ndarray, list[float], list[float | None] | None], np.ndarray]:
    """
    Compile a multivariate kernel by taking the product of 1D kernels for each variable.

    This function allows for specifying different kernels and bandwidths for each variable/dimension,
    when performing multivariate kernel-based estimation.

    **Input kernels**

    The input `kernels` is a list of 1D scaled kernel functions, one for each variable/dimension.
    Each kernel function should take in an array of distances and a bandwidth, and return
    the scaled kernel values (see the `scaled_kernel` decorator for how to create such functions
    from basic kernel definitions).

    **Input to the resulting multivariate kernel function**

    The resulting multivariate kernel function will take in an array of differences along each dimension,
    where each row corresponds to the difference between a pair of points along each dimension, and
    a list of bandwidths for each variable/dimension. The function will evaluate the product of the kernel
    evaluations for each variable, using the specified kernels and bandwidths.

    Parameters
    ----------
    kernels
        List of 1D kernel functions, one for each variable/dimension.

    Returns
    -------
        A function that returns the product of the kernel evaluations for each variable.
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
