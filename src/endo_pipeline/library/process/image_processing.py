"""Methods for processing images."""

from typing import Literal

import dask.array as da
import numpy as np
from bioio import BioImage
from skimage import exposure

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.io import load_image
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings import LOG_EPSILON
from endo_pipeline.settings.image_data import CAMERA_OFFSET


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Rescale and convert input array to uint8."""

    low, high = image.min(), image.max()
    image = exposure.rescale_intensity(image, in_range=(low, high), out_range=(0, 255))
    return image.astype(np.uint8)


def load_egfp_image(
    config: DatasetConfig, position: int, timepoints: int | list[int], level: int = 0
) -> da.Array:
    """Load EGFP max projection image for given timepoint(s)."""

    location = get_zarr_location_for_position(config, position=position)
    image = load_image(location, channels=["EGFP"], timepoints=timepoints, level=level)

    # Compute max projection along z axis
    image = image.max(axis=2)

    # Compute percentiles and clip
    low = da.percentile(image, 10, axis=(2, 3), keepdims=True)
    high = da.percentile(image, 98, axis=(2, 3), keepdims=True)
    image = da.clip(image, low, high)

    # Normalize between -1 and 1
    return (image - low) / (high - low + 1e-8) * 2.0 - 1.0


def load_bf_image(
    config: DatasetConfig, position: int, timepoints: int | list[int], level: int = 0
) -> da.Array:
    """Load BF single focal plane image for given timepoint(s)."""

    if config.center_z_plane is None:
        raise ValueError("'center_z_plane' is None, cannot load single focal plane for BF channel")

    location = get_zarr_location_for_position(config, position=position)
    image = load_image(location, channels=["BF"], timepoints=timepoints, level=level)

    # Compute visualization plane from focal plane
    focal_plane = config.center_z_plane[position]
    visualize_plane = focal_plane - 5
    image = image[:, :, visualize_plane, :, :]

    # Compute percentiles and clip
    low = da.percentile(image, 1, axis=(2, 3), keepdims=True)
    high = da.percentile(image, 99, axis=(2, 3), keepdims=True)
    image = da.clip(image, low, high)

    # Normalize between 0 and 255
    return (image - low) / (high - low + 1e-8) * 255


def load_bf_std_dev_image(
    config: DatasetConfig, position: int, timepoints: int | list[int], level: int = 0
) -> da.Array:
    """Load BF log standard deviation projection image for given timepoint(s)."""

    location = get_zarr_location_for_position(config, position=position)
    image = load_image(location, channels=["BF"], timepoints=timepoints, level=level)

    # Compute std projection along z axis and apply log transform
    image = image.std(axis=2)
    image = da.log(image + LOG_EPSILON)

    # Compute percentiles and clip
    low = da.percentile(image, 0.1, axis=(2, 3), keepdims=True)
    high = da.percentile(image, 99.9, axis=(2, 3), keepdims=True)
    image = da.clip(image, low, high)

    # Compute mean and std per timepoint and z-score normalize
    mean = image.mean(axis=(2, 3), keepdims=True)
    std = image.std(axis=(2, 3), keepdims=True)
    return (image - mean) / std


def bf_slice(img: BioImage, frame: int) -> np.ndarray:
    """
    Get the best Z slice from the brightfield image for a given frame.

    Selects the Z plane with the lowest standard deviation (in-focus plane)
    and shifts 5 planes down to obtain a plane with visible contrast.

    Parameters
    ----------
    img
        The input BioImage object containing the image data.
    frame
        The timepoint index to retrieve.

    Returns
    -------
        The selected 2D brightfield plane as a NumPy array.
    """
    bf_stack = img.get_image_dask_data("ZYX", C=1, T=frame)
    stdevs = [plane.std().compute() for plane in bf_stack.squeeze()]
    best_plane = max(0, np.argmin(stdevs) - 5)  # move 5 planes down to have contrast
    bf_slice = img.get_image_dask_data("YX", Z=best_plane, C=1, T=frame)
    return bf_slice.compute()


def bf_std_dev(img: BioImage, frame: int) -> np.ndarray:
    """
    Calculate the standard deviation projection of the brightfield image for a given frame.

    Computes the standard deviation across the Z-axis for channel 1 (brightfield).

    Parameters
    ----------
    img
        The input BioImage object containing the image data.
    frame
        The timepoint index to retrieve.

    Returns
    -------
        The 2D standard deviation projection as a NumPy array.
    """
    bf_img = img.get_image_dask_data("ZYX", C=1, T=frame, ddof=1)
    bf_std_dev = bf_img.std(axis=0)
    return bf_std_dev.compute()


def gfp_max_proj(img: BioImage, frame: int) -> np.ndarray:
    """
    Get the maximum intensity projection of the GFP channel for a given frame.

    Computes the max projection across the Z-axis for channel 0 (GFP).

    Parameters
    ----------
    img
        The input BioImage object containing the image data.
    frame
        The timepoint index to retrieve.

    Returns
    -------
        The 2D maximum intensity projection as a NumPy array.
    """
    gfp = img.get_image_dask_data("ZYX", C=0, T=frame)
    gfp_max_proj = gfp.max(axis=0)
    return gfp_max_proj.compute()


def get_single_bf_plane(stack: da.Array, offset: int = -5) -> np.ndarray:
    """
    Get a single Z plane from the brightfield stack to visualize.

    Identifies the in-focus plane (lowest standard deviation) and applies an
    offset to select a nearby plane with more visible contrast.

    Parameters
    ----------
    stack
        The brightfield Z-stack as a Dask array.
    offset
        Number of planes to shift from the in-focus plane, by default -5.

    Returns
    -------
        The selected 2D brightfield plane as a NumPy array.
    """
    stdevs = [plane.std().compute() for plane in stack.squeeze()]
    focus_plane = np.argmin(stdevs)  # in focus plane has low contrast in BF
    plane_selection = max(0, focus_plane + offset)  # shift from focus plane to get contrast
    bf_plane = stack[plane_selection]
    return bf_plane.compute()


def max_proj_405(img: BioImage, frame: int) -> np.ndarray:
    """
    Get the maximum intensity projection of the NucViolet (405 nm) channel for a given frame.

    Computes the max projection across the Z-axis for channel 2 (NucViolet).

    Parameters
    ----------
    img
        The input BioImage object containing the image data.
    frame
        The timepoint index to retrieve.

    Returns
    -------
        The 2D maximum intensity projection as a NumPy array.
    """
    channel_405 = img.get_image_dask_data("ZYX", C=2, T=frame)
    channel_405_max_proj = channel_405.max(axis=0)
    return channel_405_max_proj.compute()


def max_proj_561(img: BioImage, frame: int) -> np.ndarray:
    """
    Get the maximum intensity projection of the SOX17 (561 nm) channel for a given frame.

    Computes the max projection across the Z-axis for channel 3 (SOX17).

    Parameters
    ----------
    img
        The input BioImage object containing the image data.
    frame
        The timepoint index to retrieve.

    Returns
    -------
        The 2D maximum intensity projection as a NumPy array.
    """
    channel_561 = img.get_image_dask_data("ZYX", C=3, T=frame)
    channel_561_max_proj = channel_561.max(axis=0)
    return channel_561_max_proj.compute()


def max_proj_640(img: BioImage, frame: int) -> np.ndarray:
    """
    Get the maximum intensity projection of the SMAD1 or NR2F2 (640 nm) channel for a given frame.

    Computes the max projection across the Z-axis for channel 4.

    Parameters
    ----------
    img
        The input BioImage object containing the image data.
    frame
        The timepoint index to retrieve.

    Returns
    -------
        The 2D maximum intensity projection as a NumPy array.
    """
    channel_640 = img.get_image_dask_data("ZYX", C=4, T=frame)
    channel_640_max_proj = channel_640.max(axis=0)
    return channel_640_max_proj.compute()


def max_proj(stack: da.Array, axis: int) -> np.ndarray:
    """
    Compute the maximum intensity projection along the specified axis.

    Parameters
    ----------
    stack
        The input Dask array.
    axis
        Axis along which to compute the maximum projection.

    Returns
    -------
        The maximum intensity projection as a NumPy array.
    """
    max_proj = stack.max(axis)  # Max projection along the Z-axis
    return max_proj.compute()


def sum_proj(stack: da.Array, axis: int) -> np.ndarray:
    """
    Compute the sum projection along the specified axis.

    Parameters
    ----------
    stack
        The input Dask array.
    axis
        Axis along which to compute the sum projection.

    Returns
    -------
        The sum projection as a NumPy array.
    """
    sum_proj = stack.sum(axis)  # Sum projection along the Z-axis
    return sum_proj.compute()


def std_dev(
    stack: da.Array,
    axis: int,
    unbiased: bool = False,
) -> np.ndarray:
    """
    Compute the standard deviation projection along the specified axis on a Dask array.

    Parameters
    ----------
    stack
        The input Dask array to project.
    axis
        Axis along which to compute the standard deviation.
    unbiased
        Whether to use the unbiased estimator (ddof=1), by default False.

    Returns
    -------
        The standard deviation projection as a NumPy array.
    """
    ddof = 1 if unbiased else 0
    std_projection = stack.std(axis=axis, ddof=ddof).compute()
    return std_projection


def log_normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Apply logarithmic normalization to the input image.

    Applies a logarithmic transformation (log1p) to the pixel intensities,
    which can help enhance contrast in images with a wide dynamic range.

    Parameters
    ----------
    image
        Input image as a NumPy array.

    Returns
    -------
        Logarithmically normalized image.
    """
    # Add a small constant to avoid log(0)
    log_image = np.log1p(image)
    return log_image


def clip_image(arr: np.ndarray, low_pct: float = 0.1, high_pct: float = 99.9) -> np.ndarray:
    """
    Clip the values in a NumPy array to the specified percentiles.

    Parameters
    ----------
    arr
        Input array to be clipped.
    low_pct
        Lower percentile for clipping, by default 0.1.
    high_pct
        Upper percentile for clipping, by default 99.9.

    Returns
    -------
        The clipped array with values constrained between the specified percentiles.
    """
    low_val = np.percentile(arr, low_pct)
    high_val = np.percentile(arr, high_pct)
    return np.clip(arr, low_val, high_val)


def z_score_normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensity using z-score normalization, including zeros.

    Follows the default MONAI behavior for intensity normalization. If the
    standard deviation is zero, returns the mean-subtracted image.

    Parameters
    ----------
    image
        Input image as a NumPy array.

    Returns
    -------
        Normalized image with zero mean and unit standard deviation.
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
    Scale image intensities based on percentile range to a target range.

    Maps the intensity values between the ``lower`` and ``upper`` percentiles
    to the output range [``b_min``, ``b_max``].

    Parameters
    ----------
    img
        Input image.
    lower
        Lower percentile for the input range, by default 10.
    upper
        Upper percentile for the input range, by default 98.
    b_min
        Output minimum value, by default -1.0.
    b_max
        Output maximum value, by default 1.0.
    clip
        Whether to clip values outside [``b_min``, ``b_max``], by default True.

    Returns
    -------
        Scaled image.
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
    image_list
        List of images (numpy arrays) to compute the global range.
    method
        The method to use for calculating the range:
        - 'min-max': Use global min and max values.
        - 'percentile': Use percentiles to determine the range.

    Returns
    -------
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
    image
        The input image array.
    method
        - 'min-max': stretch between min and max
        - 'percentile': stretch between percentiles
    low_percentile
        The lower percentile for contrast stretching (default is 1).
    high_percentile
        The upper percentile for contrast stretching (default is 99).
    custom_range
        A custom range (low, high) for contrast stretching.
        If provided, overrides the method-specific ranges.
        Useful for applying the same range across multiple images.

    Returns
    -------
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


def background_subtract(
    img: np.ndarray | da.Array, camera_offset: int = CAMERA_OFFSET
) -> np.ndarray | da.Array:
    """
    Background subtract the image by clipping values below a camera offset. The camera offset
    of 100 is the value set in our hardware configuration, therefore this is the minimum value
    of intensity that should be used for analysis purposes.

    Parameters
    ----------
    img
        The input image array.
    camera_offset
        The camera offset value to subtract from the image, by default 100.

    Returns
    -------
        The background-subtracted image.
    """
    # Use da.clip for Dask arrays, np.clip for NumPy arrays
    if isinstance(img, da.Array):
        img = da.clip(img, camera_offset, None)
    else:
        img = np.clip(img, camera_offset, None)

    # Subtract the camera offset
    img = img - camera_offset
    return img


def normalize_image(image: np.ndarray, target_max: int = 255) -> np.ndarray:
    """
    Normalize the image to a specific range (e.g., 0 to ``target_max``).

    Preserves the original dynamic range of the image while scaling it to the
    target maximum value instead of clipping. If ``target_max`` is 255 or less,
    the output is cast to ``uint8``.

    Parameters
    ----------
    image
        Input image.
    target_max
        The target maximum value for normalization, by default 255.

    Returns
    -------
        Normalized image within the range [0, ``target_max``].
    """
    # Calculate the minimum and maximum pixel values of the original image
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image to the target range (0 to target_max)
    normalized_image = (image - min_val) / (max_val - min_val) * target_max

    if target_max <= 255:
        normalized_image = normalized_image.astype(np.uint8)

    return normalized_image


def crop_image(img: np.ndarray, start_x: int, start_y: int, crop_size: int) -> np.ndarray:
    """
    Crop a square region from an image array in Y, X dimensions.

    Parameters
    ----------
    img
        The input image array with shape (Y, X), (C, Y, X), (T, C, Y, X), or (T, C, Z, Y, X).
    start_x
        The starting pixel coordinate along the X-axis (horizontal) for cropping.
    start_y
        The starting pixel coordinate along the Y-axis (vertical) for cropping.
    crop_size
        The size of the square crop.

    Returns
    -------
        The cropped image region.
    """
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    slices = [slice(None)] * (img.ndim - 2) + [slice(start_y, end_y), slice(start_x, end_x)]
    return img[tuple(slices)]


def stitch_with_overlap(
    arrays: list[da.Array], overlap_ratio: float = 0.10, reference_frame: int = 0
) -> da.Array:
    """
    Stitch a list of 2D or 3D Dask arrays along the X axis with fixed linear blending in the overlapping region.

    Parameters
    ----------
    arrays
        List of tiles to stitch.
    overlap_ratio
        Fixed overlap ratio between adjacent tiles (default 1%).

    Returns
    -------
        Stitched array.
    """
    stitched = arrays[0]

    for arr in arrays[1:]:
        # Compute overlap in pixels
        overlap = max(1, int(min(stitched.shape[-1], arr.shape[-1]) * overlap_ratio))

        # Split new array
        non_overlap_new = arr[..., overlap:]
        A = stitched[..., -overlap:]
        B = arr[..., :overlap]

        # Linear blending
        wA = da.from_array(
            np.linspace(1, 0, overlap)[None, None, :]
            if stitched.ndim == 3
            else np.linspace(1, 0, overlap)
        )
        wB = 1 - wA
        blended = A * wA + B * wB

        stitched = da.concatenate([stitched[..., :-overlap], blended, non_overlap_new], axis=-1)

    return stitched
