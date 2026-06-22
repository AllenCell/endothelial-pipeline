"""Methods for processing images."""

from typing import Literal

import dask.array as da
import numpy as np
from skimage import exposure

from endo_pipeline.configs import ChannelName, DatasetConfig
from endo_pipeline.io import load_image
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings import LOG_EPSILON


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Rescale and convert input array to uint8."""

    low, high = image.min(), image.max()
    image = exposure.rescale_intensity(image, in_range=(low, high), out_range=(0, 255))
    return image.astype(np.uint8)


def load_processed_egfp_image(
    config: DatasetConfig, position: int, timepoints: int | list[int], level: int = 0
) -> da.Array:
    """Load processed EGFP max projection image."""

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


def load_processed_bf_image(
    config: DatasetConfig, position: int, timepoints: int | list[int], level: int = 0
) -> da.Array:
    """Load processed BF single focal plane image."""

    if config.center_z_plane is None:
        raise ValueError("'center_z_plane' is None, cannot load single focal plane for BF channel")

    location = get_zarr_location_for_position(config, position=position)
    image = load_image(location, channels=[ChannelName.BF], timepoints=timepoints, level=level)

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


def load_processed_bf_std_dev_image(
    config: DatasetConfig, position: int, timepoints: int | list[int], level: int = 0
) -> da.Array:
    """Load processed BF log standard deviation projection image at specified crop."""

    location = get_zarr_location_for_position(config, position=position)
    image = load_image(location, channels=[ChannelName.BF], timepoints=timepoints, level=level)

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


def load_processed_egfp_image_crop(
    config: DatasetConfig,
    position: int,
    timepoints: int | list[int],
    start_x: int,
    start_y: int,
    crop_size: int,
    level: int = 0,
) -> np.ndarray:
    """Load processed EGFP max projection image at specified crop."""

    image = load_processed_egfp_image(config, position, timepoints, level)
    return crop_image(image.squeeze().compute(), start_x, start_y, crop_size)


def load_processed_bf_image_crop(
    config: DatasetConfig,
    position: int,
    timepoints: int | list[int],
    start_x: int,
    start_y: int,
    crop_size: int,
    level: int = 0,
) -> np.ndarray:
    """Load processed BF single focal plane image at specified crop."""

    image = load_processed_bf_image(config, position, timepoints, level)
    return crop_image(image.squeeze().compute(), start_x, start_y, crop_size)


def load_processed_bf_std_dev_image_crop(
    config: DatasetConfig,
    position: int,
    timepoints: int | list[int],
    start_x: int,
    start_y: int,
    crop_size: int,
    level: int = 0,
) -> np.ndarray:
    """Load processed BF log standard deviation projection image at specified crop."""

    image = load_processed_bf_std_dev_image(config, position, timepoints, level)
    return crop_image(image.squeeze().compute(), start_x, start_y, crop_size)


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
