import logging

import dask.array as da
from bioio import BioImage

from endo_pipeline.configs import DatasetConfig

logger = logging.getLogger(__name__)


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
    position_index: int,
    dataset_config: DatasetConfig,
    channel_names: list,
    num_positions: int = 6,
    scene_index: int = 0,
    img: BioImage | None = None,
) -> da.Array:
    """
    Load selected timepoints for a given position as a Dask array.

    Every nth position (based on total number of positions) in time to create
    the Dask array.

    Parameters
    ----------
    position_index
        The position index to process.
    dataset_config
        Dataset config for the dataset to be loaded.
    channel_names
        List of channel names.
    num_positions
        Total number of positions in the dataset.
    scene_index
        Scene index.
    img
        Loaded image for the dataset. If provided, it will reduce the number of
        times the image is loaded. If not provided, it will be loaded for the
        given dataset config.

    Returns
    -------
    :
        A Dask array containing the processed images for all timepoints at the
        given position.
    """

    # Load the dataset as a BioImage object
    img = img if img is not None else BioImage(dataset_config.original_path)

    # Set the scene of the image
    img.set_scene(scene_index)
    logger.info("Using scene '%s'", img.scenes[scene_index])

    # Get the timepoints for the specified position using the position index and
    # the total number of timepoints in the image.
    t_final = img.dims.T
    timepoints = range(position_index, t_final, num_positions)

    # Get the indices of the GFP and brightfield channels
    indices = dataset_config.original_channel_indices
    channels = [indices.channel_488, indices.brightfield]

    if indices.channel_405 is not None:
        channels.append(indices.channel_405)
    if indices.channel_561 is not None:
        channels.append(indices.channel_561)
    if indices.channel_640 is not None:
        channels.append(indices.channel_640)

    if len(channels) != len(channel_names):
        raise ValueError(
            f"Number of channels '{len(channels)}' does not match "
            f"the number of channel names '{len(channel_names)}'"
        )

    # Get the delayed arrays for each timepoint at the specified position
    # with the channels in the specified order
    results = [img.get_image_dask_data("CZYX", T=tp, C=channels) for tp in timepoints]

    # Concatenate delayed arrays into a single TCZYX delayed array
    scene = da.stack(results, axis=0)
    logger.info("Loaded '%d' timepoints for position index '%d'", len(timepoints), position_index)

    return scene
