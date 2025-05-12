from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
import pandas as pd
from bioio import BioImage
from skimage import img_as_ubyte
from skimage.transform import pyramid_reduce

from cellsmap.util import dataset_io
from cellsmap.vis import get_images


def get_raw_intensity_crop(
    row: pd.Series, resolution_level: int, channel: int
) -> np.ndarray:
    p = dataset_io.extract_P(row.position)
    img = get_images.get_zarr_img_for_dataset(row.dataset, p, resolution_level)
    crop_size_x = row.crop_size_x if resolution_level == 1 else row.crop_size_x * 2
    crop_size_y = row.crop_size_y if resolution_level == 1 else row.crop_size_y * 2

    raw_crop = get_images.get_crop(
        img=img,
        channel=channel,
        timepoint=row.frame_number,
        start_x=row.start_x,
        start_y=row.start_y,
        crop_size_x=crop_size_x,
        crop_size_y=crop_size_y,
    )

    # keep z, extract one time and one channel
    raw_crop_channel = raw_crop[0, channel, :, :]
    return raw_crop_channel


def background_subtract(img: np.ndarray, camera_offset: int = 100) -> np.ndarray:
    img = np.clip(img, camera_offset, None)
    img = img - camera_offset  # Set any negative values to zero
    return img


def normalize_img(img: np.ndarray) -> np.ndarray:
    # make 8-bit
    return img_as_ubyte(img)


def total_intensity(img: np.ndarray) -> float:
    """
    Calculate the total intensity of the image.
    """
    return float(np.sum(img))


def sum_projection(img: Union[np.ndarray, da.Array]) -> np.ndarray:
    """
    Create a sum projection of the image.
    Works with both NumPy and Dask arrays and returns a NumPy array.
    """
    result = (
        np.sum(img, axis=0)
        if not hasattr(img, "compute")
        else img.sum(axis=0).compute()
    )
    return np.array(result)


def sum_projection_in_mask(sum_proj_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a sum projection of the image in the masked region.
    Returns a 2D array with masked values retained and others set to zero.
    """
    # Create a copy of the input image and set values outside the mask to zero
    result = np.zeros_like(sum_proj_img)
    result[mask] = sum_proj_img[mask]
    return result


def sum_projection_not_in_mask(
    sum_proj_img: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Create a sum projection of the image not in the masked region.
    Returns a 2D array with non-masked values retained and others set to zero.
    """
    # Create a copy of the input image and set values inside the mask to zero
    result = np.zeros_like(sum_proj_img)
    result[~mask] = sum_proj_img[~mask]
    return result


def get_segmentation_mask_crop(
    row: pd.Series,
    resolution_level: int,
    channel: Union[int, list[int]],
    binary: bool = True,
) -> np.ndarray:

    # Construct the full path to the segmentation mask
    path_to_nuc_seg = Path(
        dataset_io.get_nuclear_prediction_path(
            row.dataset, dataset_io.extract_P(row.position)
        )
    )
    fname = f"{row.dataset}_{row.position}_T{row.frame_number}.ome.tiff"
    full_path = path_to_nuc_seg / fname

    # Check if the file exists
    if not full_path.exists():
        raise FileNotFoundError(f"Segmentation mask file not found: {full_path}")

    # Load the segmentation image
    seg_img = BioImage(str(full_path))

    # Get the crop from the segmentation image
    seg_crop = get_images.get_crop(
        img=seg_img,
        channel=channel,
        timepoint=row.frame_number,
        start_x=row.start_x,
        start_y=row.start_y,
        crop_size_x=row.crop_size_x * 2,
        crop_size_y=row.crop_size_y * 2,
    )

    # # Extract channel from the crop
    seg_crop_channel = seg_crop[0, channel, 0]
    # # Remove timepoint and channel dimensions
    mask = np.squeeze(seg_crop_channel)
    # Downsample if resolution level is 1
    if resolution_level == 1:
        mask = pyramid_reduce(mask, downscale=2)

    # Apply thresholding to create a binary mask
    if binary:
        mask = mask >= 1

    if hasattr(mask, "compute"):
        mask = mask.compute()

    return np.array(mask)


def total_intensity_in_mask(
    img: Union[np.ndarray, da.Array], mask: Union[np.ndarray, da.Array]
) -> float:
    """
    Calculate the sum of pixel values in the masked region.
    """
    return float(
        np.sum(img[mask]).compute() if hasattr(img, "compute") else np.sum(img[mask])
    )


def total_intensity_not_in_mask(
    img: Union[np.ndarray, da.Array], mask: Union[np.ndarray, da.Array]
) -> float:
    """
    Calculate the sum of pixel values not in the masked region.
    """
    return float(
        np.sum(img[~mask]).compute() if hasattr(img, "compute") else np.sum(img[~mask])
    )


def mean_intensity_in_mask(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the mean intensity of the image in the masked region.
    """
    region = img[mask]
    return float(np.mean(region))


def mean_intensity_not_in_mask(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the mean intensity of the image not in the masked region.
    """
    region = img[~mask]
    return float(np.mean(region))


def median_intensity_in_mask(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the median intensity of the image in the masked region.
    """
    region = img[mask]
    return float(np.median(region))


def median_intensity_not_in_mask(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the median intensity of the image outside the masked region.
    """
    region = img[~mask]
    return float(np.median(region))
