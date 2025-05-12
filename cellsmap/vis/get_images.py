from pathlib import Path
from typing import Literal

import dask.array as da
from bioio import BioImage

from cellsmap.util import dataset_io


def get_zarr_img_for_dataset(
    dataset: str, position: int, resolution_level: Literal[0, 1] = 1
) -> BioImage:
    """
    Retrieve the BioImage object for a given dataset and position.
    """
    zarr_name = dataset_io.get_zarr_name(dataset, position)
    zarr_path = dataset_io.get_zarr_dir(dataset)
    filepath = Path(zarr_path) / zarr_name
    img = BioImage(filepath)
    img.set_resolution_level(resolution_level)
    return img


def get_timepoint(
    img: BioImage,
    timepoint: int,
) -> da.Array:
    """
    Get the image data for a specific timepoint.
    """
    img_timepoint = img.get_image_dask_data("CZYX", T=timepoint)
    return img_timepoint


def get_crop(
    img: BioImage,
    channel: int | list | None,
    timepoint: int | None,
    start_x: int,
    start_y: int,
    crop_size_x: int,
    crop_size_y: int,
) -> da.Array:
    kwargs = {
        "Y": slice(start_y, start_y + crop_size_y),
        "X": slice(start_x, start_x + crop_size_x),
        "reshape": False,
    }

    single_channel = isinstance(channel, int)
    single_time = isinstance(timepoint, int)

    if single_time:
        kwargs["T"] = slice(timepoint, timepoint + 1)
    if channel is not None:
        if single_channel:
            kwargs["channel"] = [channel]  # Use list to ensure correct indexing
        else:
            kwargs["channel"] = channel  # Accepts list or slice

    img_crop = img.get_image_dask_data("TCZYX", **kwargs)

    return img_crop
