from pathlib import Path
from typing import Union

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


def sum_projection(img: np.ndarray) -> np.ndarray:
    """
    Create a sum projection of the image.
    """
    return np.sum(img, axis=0)


def get_segmentation_mask_crop(
    row: pd.Series, resolution_level: int, channel: Union[int, list[int]]
) -> np.ndarray:

    # Construct the full path to the segmentation mask
    path_to_nuc_seg = Path(
        dataset_io.get_nuclear_prediction_path(
            row.dataset, dataset_io.extract_P(row.position)
        )
    )
    fname = f"{row.dataset}_{row.position}_T{row.frame_number}_cellpose.ome.tiff"
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
    seg_crop_YX = np.squeeze(seg_crop_channel)
    # Downsample if resolution level is 1
    if resolution_level == 1:
        seg_crop_YX = pyramid_reduce(seg_crop_YX, downscale=2)

    # Apply thresholding to create a binary mask
    binary_mask = seg_crop_YX >= 1

    return binary_mask


def sum_in_mask(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the sum of pixel values in the masked region.
    """
    return np.sum(img[mask]).compute()


def sum_not_in_mask(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the sum of pixel values not in the masked region.
    """
    return np.sum(img[~mask]).compute()


def sum_projection_in_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create an image within intensity values only in the masked region.
    """
    return img * mask


def sum_projection_not_in_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create an image within intensity values only not in the masked region.
    """
    return img * ~mask
