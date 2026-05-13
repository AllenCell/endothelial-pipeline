from typing import Any

import dask.array as da
from bioio import BioImage

from endo_pipeline.configs import DatasetConfig


def get_included_scenes(dataset_config: DatasetConfig) -> list | range:
    """
    Retrieve the list of scenes to include for a given dataset if specified
    in the dataset config file.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset for which to retrieve included scenes.

    Returns
    -------
    list
        A list of scene indices to include. If no specific scenes are defined
        in the dataset config file,
        all scenes are included by default.
    """

    include_scenes = dataset_config.include_scenes

    # by default, include all scenes
    if include_scenes is None:
        include_scenes = range(len(BioImage(dataset_config.original_path).scenes))

    return include_scenes


def get_delayed_array_for_position(
    pos: int,
    dataset_config: DatasetConfig,
    channel_names: list,
    number_positions: int = 6,
    scene_index: int = 0,
    img: Any | None = None,
) -> da.Array:
    """
    Load all timepoints for a given position in the dataset as a Dask array.

    Parameters
    ----------
    pos : int
        The position index to process.
    dataset_name : str
        The name of the dataset.
    channel_names : list
        A list of channel names.
    number_positions : int, optional
        The total number of positions in the dataset. Take every nth position in time to create
        the Dask array.
    scene_index : int, optional
        The scene index. You can find the scene names (in their indexed order)
        using `BioImage(dataset_config.original_path).scenes`. Default is 0.
    img : BioImage, optional
        The BioImage object for the dataset. If provided, it will reduce the
        number of times the image is loaded. If None, it will be loaded for
        the given dataset_name. Default is None.

    Returns
    -------
    dask.array.Array
        A Dask array containing the processed images for all timepoints at
        the given position.
    """

    # Load the dataset as a BioImage object
    img = img if img else BioImage(dataset_config.original_path)
    # Set the scene of the image
    img.set_scene(int(scene_index))
    # Get the timepoints for the specified position
    t_final = img.dims.T  #  the total number of timepoints in "img"; if testing set low t_final
    timepoints = range(pos, t_final, number_positions)
    # Get the indices of the GFP and brightfield channels

    indices = dataset_config.original_channel_indices
    channels = [indices.channel_488, indices.brightfield]

    if indices.channel_405 is not None:
        channels.append(indices.channel_405)
    if indices.channel_561 is not None:
        channels.append(indices.channel_561)
    if indices.channel_640 is not None:
        channels.append(indices.channel_640)

    assert len(channels) == len(
        channel_names
    ), f"Number of channels ({len(channels)}) does not match number of \
        channel names ({len(channel_names)})"

    # Get the delayed arrays for each timepoint at the specified position
    # with the channels in the following order: (GFP, brightfield)
    results = [img.get_image_dask_data("CZYX", T=tp, C=channels) for tp in timepoints]
    # Concatenate the delayed arrays into a single large delayed array
    # along the time axis
    scene = da.stack(results, axis=0)  # TCZYX
    print(f"finished processing {len(timepoints)} timepoints for position {pos}")
    return scene
