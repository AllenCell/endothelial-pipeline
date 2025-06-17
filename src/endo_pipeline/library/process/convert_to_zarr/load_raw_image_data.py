from typing import Any

import dask.array as da
from bioio import BioImage

from cellsmap.util.dataset_io import (
    get_dataset_info,
    get_original_path,
    get_specific_channel_order,
    get_total_number_of_positions,
)


def get_included_scenes(dataset_name: str) -> list:
    """
    Retrieves the list of scenes to include for a given dataset if specified in the dataset config file.

    Parameters:
    dataset_name (str): The name of the dataset for which to retrieve included scenes.

    Returns:
    include_scenes (list): A list of scene indices to include. If no specific scenes are defined in the dataset config file,
            all scenes are included by default.
    """
    dataset_info = get_dataset_info(dataset_name)
    include_scenes = dataset_info.get("include_scenes")

    # by default, include all scenes
    if include_scenes is None:
        include_scenes = range(len(BioImage(dataset_info["original_path"]).scenes))

    return include_scenes


def get_delayed_array_for_position(
    pos: int,
    dataset_name: str,
    channel_names: list,
    number_positions: int = 6,
    scene_index: int = 0,
    img: Any | None = None,
) -> da.Array:
    """
    Loads all timepoints for a given position in the datase as a Dask array.

    Parameters:
    pos (int): The position index to process.
    dataset_name (str): The name of the dataset.
    number_positions (int): The total number of positions in the dataset. Default is 6.
    scene_index (int): The scene index. You can find the scene names (in their indexed order) using `BioImage(get_original_path(dataset_name)).scenes` Default is 0.
    img (BioImage): The BioImage object for the dataset. If provided will reduce number of times the image is loaded. If None then it will be loaded for the given dataset_name. Default is None.

    Returns:
    dask.array.Array: A Dask array containing the processed images for all timepoints at the given position.
    """
    # Load the dataset as a BioImage object
    img = img if img else BioImage(get_original_path(dataset_name))
    # Set the scene of the image
    img.set_scene(int(scene_index))
    # Get the timepoints for the specified position
    t_final = (
        img.dims.T
    )  #  the total number of timepoints in "img"; if testing set t_final to some smaller number (recommended: 18)
    number_positions = get_total_number_of_positions(dataset_name)
    timepoints = range(pos, t_final, number_positions)
    # Get the indices of the GFP and brightfield channels
    index_488, bf_index, index_405, index_561, index_640 = get_specific_channel_order(
        dataset_name
    )

    channels = [index_488, bf_index]
    if index_405 is not None:
        channels.append(index_405)
    if index_561 is not None:
        channels.append(index_561)
    if index_640 is not None:
        channels.append(index_640)

    assert len(channels) == len(
        channel_names
    ), f"Number of channels ({len(channels)}) does not match number of channel names ({len(channel_names)})"

    # Get the delayed arrays for each timepoint at the specified position
    # with the channels in the following order: (GFP, brightfield)
    results = [img.get_image_dask_data("CZYX", T=tp, C=channels) for tp in timepoints]
    # Concatenate the delayed arrays into a single large delayed array
    # along the time axis
    scene = da.stack(results, axis=0)  # TCZYX
    print(f"finished processing {len(timepoints)} timepoints for position {pos}")
    return scene
