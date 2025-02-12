from cellsmap.util.io import (
    get_original_path,
    get_specific_channel_order,
    get_dataset_duration_in_frames,
)
import dask.delayed
import dask.array as da
import numpy as np


def get_slidebook_image_path(dataset_name: str, channel: int, timepoint: int):
    """
    Constructs the file path for a SlideBook image.

    Parameters:
    dataset_name (str): The name of the dataset.
    channel (int): The channel index.
    timepoint (int): The timepoint index.

    Returns:
    pathlib.Path: The constructed file path for the SlideBook image.
    """
    sld_path = get_original_path(dataset_name)
    sld_path = sld_path / f"ImageData_Ch{channel}_TP{timepoint:07d}.npy"
    return sld_path


def get_image_format(dataset_name: str, channel: int, timepoint: int):
    """
    Retrieves the shape and data type of a SlideBook image.

    Parameters:
    dataset_name (str): The name of the dataset.
    channel (int): The channel index.
    timepoint (int): The timepoint index.

    Returns:
    tuple: A tuple containing the shape and data type of the image.
    """
    sld_path = get_slidebook_image_path(dataset_name, channel, timepoint)
    array = np.load(sld_path)
    return array.shape, array.dtype


@dask.delayed
def delayed_np_load(filename):
    """
    Loads a NumPy array from a file in a delayed manner using Dask.

    Parameters:
    filename (str): The file path to the NumPy array. (i.e. SlideBook image)

    Returns:
    numpy.ndarray: The loaded NumPy array.
    """
    return np.load(filename)


def load_original_slidebook_image(
    dataset_name: str, channel: int, timepoint: int
) -> da.Array:
    """
    Loads a SlideBook image as a Dask array.

    Parameters:
    dataset_name (str): The name of the dataset.
    channel (int): The channel index.
    timepoint (int): The timepoint index.

    Returns:
    dask.array.Array: The loaded SlideBook image as a Dask array.
    """
    sld_path = get_slidebook_image_path(dataset_name, channel, timepoint)
    delayed_array = delayed_np_load(sld_path)
    return delayed_array


def get_delayed_array_for_timepoint(tp: int, dataset: str, shape: tuple, dtype: np.dtype) -> da.Array:
    """
    Processes a single timepoint for a given dataset. GFP and BF channels are set to 0 and 1, respectively.

    Parameters:
    tp (int): The timepoint index to process.
    dataset (str): The name of the dataset.
    shape (tuple): The shape of the images to be processed. This is required to correctly initialize the Dask arrays.
    dtype (str): The data type of the images to be processed. This is required to correctly initialize the Dask arrays.

    Returns:
    dask.array.Array: A Dask array containing the stacked GFP and BF images for the given timepoint.
    """
    gfp_index, bf_index = get_specific_channel_order(dataset)
    gfp = load_original_slidebook_image(dataset, channel=gfp_index, timepoint=tp)
    bf = load_original_slidebook_image(dataset, channel=bf_index, timepoint=tp)
    gfp_da = da.from_delayed(gfp, shape=shape, dtype=dtype)
    bf_da = da.from_delayed(bf, shape=shape, dtype=dtype)
    stack = da.stack([gfp_da, bf_da], axis=0)
    return stack


def get_timepoints_for_position(pos: int, dataset: str, number_positions: int = 6) -> range:
    """
    Generates a list of timepoints for a scene (time series of the same position) in the dataset.
    The montage collection of sldy files results in raw data that is organized
    in increments of time (t) for each position acquired. To assemble the full time series for a scene,
    we need to select timepoints in increments of the total number of positions in the dataset.

    Parameters:
    pos (int): The position index.
    dataset (str): The name of the dataset.
    number_positions (int): The total number of positions in the dataset. Default is 6.

    Returns:
    range: A range object containing the timepoints for the given position.
    """
    t_final = get_dataset_duration_in_frames(dataset)  # if testing set t_final = 10
    timepoints = range(pos, t_final * number_positions, number_positions)
    return timepoints


def get_delayed_array_for_position(pos: int, dataset: str, number_positions: int = 6) -> da.Array:
    """
    Loads all timepoints for a given position in the datase as a Dask array.

    Parameters:
    pos (int): The position index to process.
    dataset (str): The name of the dataset.
    number_positions (int): The total number of positions in the dataset. Default is 6.

    Returns:
    dask.array.Array: A Dask array containing the processed images for all timepoints at the given position.
    """
    timepoints = get_timepoints_for_position(pos, dataset, number_positions)
    shape, dtype = get_image_format(dataset, 0, 0)
    results = [get_delayed_array_for_timepoint(tp, dataset, shape, dtype) for tp in timepoints]
    scene = da.stack(results, axis=0)
    print(f"finished processing {len(timepoints)} timepoints")
    return scene
