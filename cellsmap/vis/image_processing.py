from typing import Literal

import dask.array as da
import numpy as np
from bioio import BioImage
from skimage import exposure


def bf_slice(img: BioImage, frame: int) -> np.ndarray:
    bf_stack = img.get_image_dask_data("ZYX", C=1, T=frame)
    stdevs = [plane.std().compute() for plane in bf_stack.squeeze()]
    best_plane = max(0, np.argmin(stdevs))
    bf_slice = img.get_image_dask_data("YX", Z=best_plane, C=1, T=frame)
    return bf_slice.compute()


def bf_max_proj(img: BioImage, frame: int) -> np.ndarray:
    bf_img = img.get_image_dask_data("ZYX", C=1, T=frame)
    bf_max_proj = bf_img.max(axis=0)
    return bf_max_proj.compute()


def bf_std_dev(img: BioImage, frame: int) -> np.ndarray:
    bf_img = img.get_image_dask_data("ZYX", C=1, T=frame)
    bf_std_dev = bf_img.std(axis=0)
    return bf_std_dev.compute()


def gfp_max_proj(img: BioImage, frame: int) -> np.ndarray:
    gfp = img.get_image_dask_data("ZYX", C=0, T=frame)
    gfp_max_proj = gfp.max(axis=0)
    return gfp_max_proj.compute()


def infocus_slice(bf_stack: da.Array) -> np.ndarray:
    """
    Get the best focus slice from a Dask array representing the brightfield stack.
    """
    # Calculate standard deviations lazily for each plane
    stdevs = [plane.std() for plane in bf_stack]

    # Compute the best plane index (requires computing the std devs)
    best_plane = max(
        0, np.argmin([s.compute() for s in stdevs]) - 5
    )  # Move 5 planes down

    # Return the best focus slice as a Dask array
    bf_slice = bf_stack[best_plane, :, :]
    return bf_slice.compute()


def max_proj(stack: da.Array, axis: int) -> np.ndarray:
    """
    Get the maximum projection of the brightfield stack as a Dask array.
    """
    max_proj = stack.max(axis)  # Max projection along the Z-axis
    return max_proj.compute()


def std_dev(stack: da.Array, axis: int) -> np.ndarray:
    """
    Get the standard deviation projection stack as a Dask array.
    """
    std_dev = stack.std(axis)  # Standard deviation along the Z-axis
    return std_dev.compute()


def contrast_stretching(
    image: np.ndarray,
    method: Literal["min-max", "percentile"] = "percentile",
    low_percentile: int = 1,
    high_percentile: int = 99,
) -> np.ndarray:
    """
    Apply contrast stretching to an image.

    Parameters:
    image (ndarray): The input image.
    method (str): The method of contrast stretching ('min-max' or 'percentile').
    low_percentile (int): The low percentile for percentile contrast stretching.
    high_percentile (int): The high percentile for percentile contrast stretching.

    Returns:
    ndarray: The contrast-stretched image.
    """
    if method == "min-max":
        low = image.min()
        high = image.max()
    elif method == "percentile":
        low, high = np.percentile(image, (low_percentile, high_percentile))

    stretched_image = exposure.rescale_intensity(
        image, in_range=(low, high), out_range=(0, 255)
    )
    return stretched_image
