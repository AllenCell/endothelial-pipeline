from typing import Literal

import dask.array as da
import numpy as np
from bioio import BioImage
from skimage import exposure


def bf_slice(img: BioImage, frame: int) -> np.ndarray:
    """Get the best Z slice from the brightfield image for a given frame."""
    bf_stack = img.get_image_dask_data("ZYX", C=1, T=frame)
    stdevs = [plane.std().compute() for plane in bf_stack.squeeze()]
    best_plane = max(0, np.argmin(stdevs) - 5)  # move 5 planes down to have contrast
    bf_slice = img.get_image_dask_data("YX", Z=best_plane, C=1, T=frame)
    return bf_slice.compute()


def bf_std_dev(img: BioImage, frame: int) -> np.ndarray:
    """Calculate the standard deviation of the brightfield image for a given frame."""
    bf_img = img.get_image_dask_data("ZYX", C=1, T=frame)
    bf_std_dev = bf_img.std(axis=0)
    return bf_std_dev.compute()


def gfp_max_proj(img: BioImage, frame: int) -> np.ndarray:
    """Get the maximum projection of the GFP channel for a given frame."""
    gfp = img.get_image_dask_data("ZYX", C=0, T=frame)
    gfp_max_proj = gfp.max(axis=0)
    return gfp_max_proj.compute()


def max_proj(stack: da.Array, axis: int) -> np.ndarray:
    """Get the maximum projection of the brightfield stack as a Dask array."""
    max_proj = stack.max(axis)  # Max projection along the Z-axis
    return max_proj.compute()


def std_dev(stack: da.Array, axis: int) -> np.ndarray:
    """Get the standard deviation projection stack as a Dask array."""
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

    Parameters
    ----------
    image : ndarray
        The input image for contrast stretching.
    method : str
        The method of contrast stretching ('min-max' or 'percentile').
    low_percentile : int
        The low percentile for percentile contrast stretching.
    high_percentile : int
        The high percentile for percentile contrast stretching.

    Returns
    -------
    ndarray
        The contrast-stretched image.
    """
    if method == "min-max":
        low = image.min()
        high = image.max()
    elif method == "percentile":
        low, high = np.percentile(image, (low_percentile, high_percentile))

    stretched_image = exposure.rescale_intensity(image, in_range=(low, high), out_range=(0, 255))
    return stretched_image
