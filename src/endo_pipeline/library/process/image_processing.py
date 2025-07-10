from typing import Literal

import dask.array as da
import numpy as np
from bioio import BioImage
from skimage import exposure

CAMERA_OFFSET = 100  # Camera offset value set in our hardware configuration


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


def get_single_bf_plane(stack: da.Array) -> np.ndarray:
    """Get a single best Z plane from the brightfield stack."""
    stdevs = [plane.std().compute() for plane in stack.squeeze()]
    best_plane = max(0, np.argmin(stdevs) - 5)  # move 5 planes down to have contrast
    bf_plane = stack[best_plane]
    return bf_plane.compute()


def max_proj(stack: da.Array, axis: int) -> np.ndarray:
    """Get the maximum projection of the brightfield stack as a Dask array."""
    max_proj = stack.max(axis)  # Max projection along the Z-axis
    return max_proj.compute()


def sum_proj(stack: da.Array, axis: int) -> np.ndarray:
    """Get the sum projection of the brightfield stack as a Dask array."""
    sum_proj = stack.sum(axis)  # Sum projection along the Z-axis
    return sum_proj.compute()


def std_dev(stack: da.Array, axis: int) -> np.ndarray:
    """Get the standard deviation projection stack as a Dask array."""
    std_dev = stack.std(axis)  # Standard deviation along the Z-axis
    return std_dev.compute()


def get_global_custom_range(
    image_list: list[np.ndarray], method: Literal["min-max", "percentile"] = "percentile"
) -> tuple[float, float]:
    """Get the global minimum and maximum values across a list of images
    for use in contrast stretching.

    Parameters
    ----------
    image_list : list[np.ndarray]
        List of images (numpy arrays) to compute the global range.
    method : str
        The method to use for calculating the range:
        - 'min-max': Use global min and max values.
        - 'percentile': Use percentiles to determine the range.

    Returns
    -------
    tuple[float, float]
        The global minimum and maximum values for contrast stretching.
    """
    if method == "min-max":
        low = min(image.min() for image in image_list)
        high = max(image.max() for image in image_list)

    elif method == "percentile":
        low = np.percentile(np.concatenate([image.flatten() for image in image_list]), 1)
        high = np.percentile(np.concatenate([image.flatten() for image in image_list]), 99)

    else:
        raise ValueError(f"Unsupported method: {method}")

    return low, high


def contrast_stretching(
    image: np.ndarray,
    method: Literal["min-max", "percentile"] = "percentile",
    low_percentile: int = 1,
    high_percentile: int = 99,
    custom_range: tuple[float, float] = None,
) -> np.ndarray:
    """
    Contrast stretching with selectable method.

    Parameters
    ----------
    image : np.ndarray
        The input image array.
    method : str, optional
        - 'min-max': stretch between min and max
        - 'percentile': stretch between percentiles
    low_percentile : int, optional
        The lower percentile for contrast stretching (default is 1).
    high_percentile : int, optional
        The upper percentile for contrast stretching (default is 99).
    custom_range : tuple, optional
        A custom range (low, high) for contrast stretching.
        If provided, overrides the method-specific ranges.
        Useful for applying the same range across multiple images.

    Returns
    -------
    np.ndarray
        The contrast-stretched image as an 8-bit unsigned integer array.
    """
    if method == "min-max":
        low, high = image.min(), image.max()

    elif method == "percentile":
        low, high = np.percentile(image, (low_percentile, high_percentile))

    else:
        raise ValueError(f"Unsupported method: {method}")

    if custom_range is not None:
        low, high = custom_range

    stretched = exposure.rescale_intensity(image, in_range=(low, high), out_range=(0, 255))
    return stretched.astype(np.uint8)


def background_subtract(img: np.ndarray, camera_offset: int = CAMERA_OFFSET) -> np.ndarray:
    """
    Background subtract the image by clipping values below a camera offset. The camera offset
    of 100 is the value set in our hardware configuration, therefore this is the minimum value
    of intensity that should be used for analysis purposes.

    Parameters
    ----------
    img : np.ndarray
        The input image array.
    camera_offset : int, optional
        The camera offset value to subtract from the image, by default 100.

    Returns
    -------
    np.ndarray
        The background-subtracted image.
    """
    img = np.clip(img, camera_offset, None)
    img = img - camera_offset  # Set any negative values to zero
    return img


def normalize_image(image: np.ndarray, target_max: int = 255) -> np.ndarray:
    """
    Normalize the image to a specific range (e.g., 0 to 255).
    8-bit images typically have pixel values in the range of 0 to 255, but this method preserves
    the original dynamic range of the image while scaling it to the target maximum value instead
    of clipping the values.

    Args:
        image (np.ndarray): Input image.
        target_max (int): The target maximum value for normalization (e.g., 255 for 8-bit).

    Returns:
        np.ndarray: Normalized image within the target range.
    """
    # Calculate the minimum and maximum pixel values of the original image
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image to the target range (0 to target_max)
    normalized_image = (image - min_val) / (max_val - min_val) * target_max

    if normalized_image.max() <= 255:
        normalized_image = normalized_image.astype(np.uint8)

    return normalized_image
