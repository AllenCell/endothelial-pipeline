from typing import Any, Tuple
from pathlib import Path
from bioio import BioImage
from cellsmap.util.io import (
    get_original_path,
    get_specific_channel_order,
    get_dataset_duration_in_frames,
    get_number_of_positions,
)
import dask.array as da


def get_slidebook_image_path(dataset_name: str) -> Path:
    """
    Constructs the file path for a SlideBook image.

    Parameters:
    dataset_name (str): The name of the dataset.
    channel (int): The channel index.
    timepoint (int): The timepoint index.

    Returns:
    pathlib.Path: The constructed file path for the SlideBook image.
    """
    sldy_path = get_original_path(dataset_name)
    return sldy_path


def get_timepoints_for_position(
    pos: int, dataset_name: str, number_positions: int = 6
) -> range:
    """
    Generates a list of timepoints for a scene (time series of the same position) in the dataset.
    The montage collection of sldy files results in raw data that is organized
    in increments of time (t) for each position acquired. To assemble the full time series for a scene,
    we need to select timepoints in increments of the total number of positions in the dataset.

    Parameters:
    pos (int): The position index.
    dataset_name (str): The name of the dataset.
    number_positions (int): The total number of positions in the dataset. Default is 6.

    Returns:
    range: A range object containing the timepoints for the given position.
    """
    t_final = get_dataset_duration_in_frames(dataset_name)  # if testing set t_final = 10
    timepoints = range(pos, t_final * number_positions, number_positions)
    return timepoints


def get_delayed_array_for_position(
    pos: int, dataset_name: str, number_positions: int = 6, scene_index: int = 0,
) -> da.Array:
    """
    Loads all timepoints for a given position in the datase as a Dask array.

    Parameters:
    pos (int): The position index to process.
    dataset_name (str): The name of the dataset.
    number_positions (int): The total number of positions in the dataset. Default is 6.

    Returns:
    dask.array.Array: A Dask array containing the processed images for all timepoints at the given position.
    """
    # Load the dataset as a BioImage object
    img = BioImage(get_original_path(dataset_name))
    # Set the scene of the image
    img.set_scene(int(scene_index))
    # Get the timepoints for the specified position
    t_final = img.dims.T #  the total number of timepoints in "img"
    number_positions = get_number_of_positions(dataset_name)
    timepoints = range(pos, t_final, number_positions)
    # Get the indices of the GFP and brightfield channels
    gfp_index, bf_index = get_specific_channel_order(dataset_name)
    # Get the delayed arrays for each timepoint at the specified position
    # with the channels in the following order: (GFP, brightfield)
    results = [img.get_image_dask_data('CZYX', T=tp, C=[gfp_index, bf_index]) for tp in timepoints]
    # Concatenate the delayed arrays into a single large delayed array
    # along the time axis
    scene = da.stack(results, axis=0)  # TCZYX
    print(f"finished processing {len(timepoints)} timepoints")
    return scene
