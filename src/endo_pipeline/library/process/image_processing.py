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
    bf_img = img.get_image_dask_data("ZYX", C=1, T=frame, ddof=1)
    bf_std_dev = bf_img.std(axis=0)
    return bf_std_dev.compute()


def gfp_max_proj(img: BioImage, frame: int) -> np.ndarray:
    """Get the maximum projection of the GFP channel for a given frame."""
    gfp = img.get_image_dask_data("ZYX", C=0, T=frame)
    gfp_max_proj = gfp.max(axis=0)
    return gfp_max_proj.compute()


def get_single_bf_plane(stack: da.Array, offset: int = -5) -> np.ndarray:
    """Get a single Z plane from the brightfield stack to visualize."""
    stdevs = [plane.std().compute() for plane in stack.squeeze()]
    focus_plane = np.argmin(stdevs)  # in focus plane has low contrast in BF
    plane_selection = max(0, focus_plane + offset)  # shift from focus plane to get contrast
    bf_plane = stack[plane_selection]
    return bf_plane.compute()


def max_proj_405(img: BioImage, frame: int) -> np.ndarray:
    """Get the maximum projection of the NucViolet channel for a given frame."""
    channel_405 = img.get_image_dask_data("ZYX", C=2, T=frame)
    channel_405_max_proj = channel_405.max(axis=0)
    return channel_405_max_proj.compute()


def max_proj_561(img: BioImage, frame: int) -> np.ndarray:
    """Get the maximum projection of the SOX17 channel for a given frame."""
    channel_561 = img.get_image_dask_data("ZYX", C=3, T=frame)
    channel_561_max_proj = channel_561.max(axis=0)
    return channel_561_max_proj.compute()


def max_proj_640(img: BioImage, frame: int) -> np.ndarray:
    """Get the maximum projection of the SMAD1 or NR2F2 channel for a given frame."""
    channel_640 = img.get_image_dask_data("ZYX", C=4, T=frame)
    channel_640_max_proj = channel_640.max(axis=0)
    return channel_640_max_proj.compute()


def max_proj(stack: da.Array, axis: int) -> np.ndarray:
    """Get the maximum projection of the brightfield stack as a Dask array."""
    max_proj = stack.max(axis)  # Max projection along the Z-axis
    return max_proj.compute()


def sum_proj(stack: da.Array, axis: int) -> np.ndarray:
    """Get the sum projection of the brightfield stack as a Dask array."""
    sum_proj = stack.sum(axis)  # Sum projection along the Z-axis
    return sum_proj.compute()


def std_dev(
    stack: da.Array,
    axis: int,
    unbiased: bool = False,
) -> np.ndarray:
    """
    Compute the standard deviation projection along the specified axis on a Dask array.

    Args:
        stack: Dask array to project.
        axis: Axis along which to compute std.
        unbiased: Whether to use unbiased estimator (ddof=1). Default False.

    Returns:
        NumPy ndarray with std projection.
    """
    ddof = 1 if unbiased else 0
    std_projection = stack.std(axis=axis, ddof=ddof).compute()
    return std_projection


def clip_image(arr: np.ndarray, low_pct: float = 0.1, high_pct: float = 99.9) -> np.ndarray:
    """
    Clip the values in a NumPy array to the specified percentiles.

    Args:
        arr (np.ndarray): Input array to be clipped.
        low_pct (float): Lower percentile for clipping (default is 0.1).
        high_pct (float): Upper percentile for clipping (default is 99.9).

    Returns:
        np.ndarray: The clipped array with values constrained between the specified percentiles.
    """
    low_val = np.percentile(arr, low_pct)
    high_val = np.percentile(arr, high_pct)
    return np.clip(arr, low_val, high_val)


def z_score_normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Normalize intensity including zeros (default MONAI behavior).

    Args:
        image: Input NumPy array.

    Returns:
        Normalized image with zero mean and unit std.
    """
    mean = image.mean()
    std = image.std()

    # Avoid division by zero
    if std == 0:
        return image - mean

    normalized = (image - mean) / std
    return normalized


def scale_intensity_range_percentiles(
    img: np.ndarray,
    lower: float = 10,
    upper: float = 98,
    b_min: float = -1.0,
    b_max: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Scale image intensities based on percentile range to a target range [b_min, b_max].

    Args:
        img (np.ndarray): Input image.
        lower (float): Lower percentile (e.g., 10).
        upper (float): Upper percentile (e.g., 98).
        b_min (float): Output minimum value.
        b_max (float): Output maximum value.
        clip (bool): Whether to clip values outside [b_min, b_max].

    Returns:
        image (np.ndarray): Scaled image.
    """
    p_low = np.percentile(img, lower)
    p_high = np.percentile(img, upper)

    scaled = (img - p_low) / (p_high - p_low) * (b_max - b_min) + b_min
    if clip:
        scaled = np.clip(scaled, b_min, b_max)

    return scaled


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
    custom_range: tuple[float, float] | None = None,
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
    if custom_range is not None:
        low, high = custom_range
    else:
        if method == "min-max":
            low, high = image.min(), image.max()
        elif method == "percentile":
            low, high = np.percentile(image, (low_percentile, high_percentile))
        else:
            raise ValueError(f"Unsupported method: {method}")

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

    if target_max <= 255:
        normalized_image = normalized_image.astype(np.uint8)

    return normalized_image
