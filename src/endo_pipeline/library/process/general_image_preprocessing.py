import logging
from collections.abc import Callable, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from bioio.writers import OmeTiffWriter
from tqdm import tqdm

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.settings.image_data import DIMENSION_ORDER

logger = logging.getLogger(__name__)


class ImageProcessingArgs(NamedTuple):
    """Structure for image processing arguments."""

    dataset_name: str
    output_dir: Path
    position: int
    timepoint: int
    img_bin_level: int
    save_output: bool
    is_validation_image: bool
    overwrite: bool


def build_analysis_queue(
    dataset_names: list,
    t_start: int = 0,
    t_final: int | None = None,
    t_step: int = 1,
    img_bin_level: int = 0,
    save_output: bool = True,
    overwrite: bool = False,
    out_dir: str | Path | None = None,
    image_validation_frequency: int | None = None,
    max_positions: int | None = None,
) -> list[ImageProcessingArgs]:
    """
    Build a list of argument tuples to be passed to image processing methods.

    Convenient for multiprocessing directly or the resulting analysis queue can
    be turned in to a pandas dataframe which can then be grouped with `groupby`
    and those groups can be passed to multiprocessing. Can also be iterated
    through with a regular for-loop. The parameters that this function takes are
    what will be included in each dictionary in the analysis queue as arguments
    to be passed to a function.

    Parameters
    ----------
    dataset_names
        A list of dataset names to build the analysis queue for.
    t_start
        Starting timepoint to analyze.
    t_final
        Final timepoint to analyze. If not provided, analyze all timepoints.
    t_step
        The step size between timepoint to analyze.
    img_bin_level:
        Image binning level to use when loading images.
    save_output:
        True to save analysis output, False otherwise.
    overwrite:
        True overwrite existing output files, False otherwise.
    out_dir:
        Output directory for analysis results. If not provided, a temporary
        output directory will be created.
    image_validation_frequency
        Frequency at which to create validation images. If not provided, no
        validation images will be created.
    max_positions
        Maximum number of positions to analyze. If not provided, analyze all
        positions.

    Returns
    -------
    :
        A list of argument tuples for each image to be analyzed.
    """

    logger.info(f"Building analysis queue for the following datasets: {dataset_names}")

    analysis_queue: list = []
    out_dir = (
        Path(out_dir) if out_dir is not None else get_output_path("analysis_queue_output_temp")
    )

    for dataset_name in dataset_names:
        # Load the dataset config
        dataset_config = load_dataset_config(dataset_name)

        # Get list of positions for the dataset. If given, limit number of
        # positions to the specified number
        position_list = dataset_config.zarr_positions
        if max_positions is not None:
            position_list = position_list[:max_positions]

        # Get range of timepoints to be evaluated
        t_final_as_int = dataset_config.duration if t_final is None else t_final
        t_range = range(t_start, t_final_as_int, t_step)

        # Get range of timepoints for validation images
        if image_validation_frequency is not None:
            validation_t_range = range(t_start, t_final_as_int, image_validation_frequency)
        else:
            validation_t_range = range(0)  # empty range will produce empty list

        for position in position_list:
            for timepoint in t_range:
                is_validation_image = True if timepoint in validation_t_range else False

                # Build argument tuple for each timeframe to be analyzed
                analysis_args = ImageProcessingArgs(
                    dataset_name=dataset_name,
                    output_dir=out_dir,
                    timepoint=timepoint,
                    position=position,
                    img_bin_level=img_bin_level,
                    save_output=save_output,
                    is_validation_image=is_validation_image,
                    overwrite=overwrite,
                )

                analysis_queue.append(analysis_args)

    return analysis_queue


def run_task_queue_with_multiprocessing(
    task: Callable,
    queue: list[ImageProcessingArgs],
    description: str,
    num_processes: int,
    chunksize: int,
) -> None:
    """
    Process tasks in queue with multiprocessing.

    Parameters
    ----------
    task
        Method to be called for each task in queue.
    queue
        List of image processing arguments for each task.
    description
        Description for the progress bar.
    num_processes
        Number of processes to use.
    chunksize
        Number of items from queue to send to each process.
    """

    logger.info("Starting multiprocessing...")
    with Pool(processes=num_processes) as pool:
        list(
            tqdm(
                pool.imap(task, queue, chunksize=chunksize),
                desc=f"{description} (MP)",
                total=len(queue),
            )
        )


def run_task_queue_in_series(
    task: Callable, queue: list[ImageProcessingArgs], description: str
) -> None:
    """
    Process tasks in queue in series.

    Parameters
    ----------
    task
        Method to be called for each task in queue.
    queue
        List of image processing arguments for each task.
    description
        Description for the progress bar.
    """

    logger.info("Starting single-core processing...")
    for item in tqdm(queue, desc=f"{description} (1P)", total=len(queue)):
        task(item)


def process_task_queue(
    task: Callable,
    queue: list,
    description: str,
    num_processes: int,
    chunksize: int,
) -> None:
    """
    Process tasks in queue in series or with multiprocessing.

    If requesting more than one process, use multiprocessing. Otherwise, process
    the task queue in series.

    Parameters
    ----------
    task
        Method to be called for each task in queue.
    queue
        List of image processing arguments for each task.
    description
        Description for the progress bar.
    num_processes
        Number of processes to use.
    chunksize
        Number of items from queue to send to each process.
    """

    if num_processes > 1:
        run_task_queue_with_multiprocessing(task, queue, description, num_processes, chunksize)
    else:
        run_task_queue_in_series(task, queue, description)


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
        dim in list(full_dims) for dim in list(current_dims)
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
            'physical_pixel_sizes': PhysicalPixelSize object (Z, Y, X)
                the physical pixel sizes as a PhysicalPixelSize object
            'dim_order': string
                the order of the dimensions of the arrays in images (e.g. 'CYX')

    Returns
    -------
    Nothing (saves an image to out_path).
    """

    assert all(
        img.shape == images[-1].shape for img in images
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
