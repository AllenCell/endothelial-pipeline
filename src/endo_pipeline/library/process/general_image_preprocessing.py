import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from tqdm import tqdm

from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.settings import DIMENSION_ORDER

logger = logging.getLogger(__name__)


def get_chan_map(filepath: Path) -> dict:
    img = BioImage(filepath)
    return {name: index for index, name in enumerate(img.channel_names)}


def build_analysis_queue(
    dataset_name_list: list,
    t_start: int = 0,
    t_final: int | None = None,
    t_step: int = 1,
    img_bin_level: int = 0,
    save_output: bool = True,
    overwrite: bool = False,
    out_dir: str | Path | None = None,
    image_validation_frequency: int | None = None,
    verbose: bool = False,
    is_test: bool = False,
) -> list:

    logger.info(f"Building analysis queue for the following datasets: {dataset_name_list}")

    analysis_queue: list = []
    out_dir = Path(out_dir) if out_dir != None else get_output_path("analysis_queue_output_temp")
    for dataset_name in tqdm(
        dataset_name_list,
        total=len(dataset_name_list),
        desc="Building analysis queue",
        unit="dataset",
    ):
        # load the dataset config
        dataset_config = load_dataset_config(dataset_name)

        # get the timeframes of the timelapse to be evaluated
        if t_final is None:
            t_final = dataset_config.duration
        t_range = range(t_start, t_final, t_step)

        # get the timeframes to be used for validation images, if any
        if image_validation_frequency is not None:
            validation_t_range = range(t_start, t_final, image_validation_frequency)
        else:
            validation_t_range = range(0)  # empty range will produce empty list

        # get the filepaths for each position in the timelapse
        for position in dataset_config.zarr_positions:
            zarr_file = get_zarr_file_for_position(dataset_config, position)

            # build a dictionary with the analysis arguments for each timeframe to be analyzed
            for timepoint in t_range:
                validation_image = True if timepoint in validation_t_range else False

                analysis_args = {
                    "dataset_name": dataset_name,
                    "image_bin_level": img_bin_level,
                    "position": position,
                    "T": timepoint,
                    "input_path": zarr_file.as_posix(),
                    "output_dir": out_dir,
                    "save_output": save_output,
                    "overwrite": overwrite,
                    "is_validation_image": validation_image,
                    "image_validation_frequency": image_validation_frequency,
                    "is_test": is_test,
                    "verbose": verbose,
                }

                analysis_queue.append(analysis_args)

    return analysis_queue


def sequence_to_scalar(sequence_like: Sequence | pd.Series) -> Any:
    """
    Takes a sequence-like object and returns the sole element if
    there is only one unique element in the sequence, else raises
    an error.
    Useful for turning a list with only 1 unique element into that
    element.
    Beware that this will return the key, not the value, if a
    dictionary is passed in.

    Example 1:
        >>> sequence_to_scalar([1])
        1
    Example 2:
        >>> sequence_to_scalar(np.array(['a', 'a', 'a'])
        'a'
    Example 3:
        >>> sequence_to_scalar([1.0, 1.1])
        ValueError: Sequence must have only one unique element.
    """
    unique_elements = set(sequence_like)
    if len(unique_elements) == 1:
        element = unique_elements.pop()
    else:
        raise ValueError(
            "Sequence must have only one unique element. " f"Unique elements: {unique_elements}"
        )

    return element


def restore_full_dims(
    image: np.ndarray, current_dims: str, full_dims: str = DIMENSION_ORDER
) -> np.ndarray:
    """
    Takes an array with specified image dims and restores dimensions with size 1
    that are present in full_dims.
    Useful for saving images using BIOIO such that they have the full TCZYX
    dimensionality so that they will load properly when opened with ImageJ/FIJI.

    NOTE: the letters in current_dims and full_dims are case sensitive and their
    order matters.

    Parameters
    ----------
    image: np.array
        The image to restore the dimensions of.

    current_dims: str
        The dimensions of 'image' as a string. Possible dimensions are:
            S: scene / position
            T: time
            C: channel
            Z: z position
            Y: y position
            X: x position

    full_dims: str
        The dimensions to restore the image to.

    Returns
    -------
    image: np.ndarray
        The image with its dimensions expanded.
    """

    assert all(
        [dim in list(full_dims) for dim in list(current_dims)]
    ), "All dimensions in current_dims must be in full_dims."

    for dim in full_dims:
        if dim not in list(current_dims):
            image = np.expand_dims(image, axis=full_dims.index(dim))

    return image


def save_image_output(
    out_path: str | Path,
    images: list[np.ndarray],
    images_metadata: dict,
    dtype: Any | None = None,
) -> None:
    """
    Combines a list of images into a single image and saves it as an OME-TIFF
    along with metadata using bioio.OmeTiffWriter.save().

    Parameters
    ----------
    out_path: Path

    images: list
        A list of numpy arrays

    images_metadata: dict
        The metadata to pass along to OmeTiffWrite.save().
        Requires the following keys:
            'image_name': string
            'channel_colors': [(int, int, int) , (int, int, int), ... )]
                where each tuple is a color defined in RGB and the length of channel_colors
                is the same length as the length of images
            'channel_names': [string, string, ...]
                where each string is a channel name and channel_names is the same length as
                the length of images
            'physical_pixel_sizes': (Z, Y, X)
                the physical pixel sizes in the order Z, Y, X.
            'dim_order': string
                the order of the dimensions of the arrays in images (e.g. 'CYX')

    Returns
    -------
    Nothing (saves an image to out_path).
    """

    assert all(
        [img.shape == images[-1].shape for img in images]
    ), "All images must have the same shape."
    # if a data type is not specified then use the smallest uint type that can hold the max value
    # among all images being saved
    if not dtype:
        img_max = max([img.max() for img in images])
        dtypes = {
            np.iinfo(dtype).max: dtype
            for dtype in (np.uint8, np.uint16, np.uint32)
            if img_max <= np.iinfo(dtype).max
        }

        assert dtypes, """
        Max pixel value in one of the channels to be saved exceeds uint32 data type, unable to save OME-TIFFs with dtype uint64 of greater.
        Please find a way to reduce the max value in the culprit channel or save the image in a different format.
        """

        dtype = dtypes[min(dtypes)]
    else:
        pass

    image_name = images_metadata["image_name"]
    ch_colors = images_metadata["channel_colors"]
    ch_names = images_metadata["channel_names"]
    px_res = images_metadata["physical_pixel_sizes"]
    img_dim_order = images_metadata["dim_order"]

    merged_img = np.concatenate(
        [restore_full_dims(img, img_dim_order, full_dims=DIMENSION_ORDER) for img in images],
        axis=DIMENSION_ORDER.index("C"),
    ).astype(dtype)

    OmeTiffWriter.save(
        merged_img,
        out_path,
        physical_pixel_sizes=px_res,
        dim_order=DIMENSION_ORDER,
        image_name=image_name,
        channel_names=ch_names,
        channel_colors=ch_colors,
    )
