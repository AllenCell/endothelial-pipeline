import logging

from bioio import BioImage

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.library.process.convert_to_zarr.load_raw_image_data import (
    get_delayed_array_for_position,
)
from endo_pipeline.library.process.convert_to_zarr.write_zarr import (
    get_sldy_pixel_sizes,
    write_scene,
)

logger = logging.getLogger(__name__)


def convert_dataset(
    dataset_config: DatasetConfig,
    output_path: str,
    channel_names: list[str],
    max_timepoints: int | None = None,
    max_positions: int | None = None,
) -> None:
    """
    Convert raw dataset to Zarr format with specific channel order.

    Parameters
    ----------
    dataset_config
        Dataset config for the dataset to be converted.
    output_path
        Base directory where the converted Zarr files will be saved.
    channel_names
        List of channel names to include in the output Zarr.
    max_timepoints
        Maximum number of timepoints to convert. Defaults to dataset duration.
    max_positions
        Maximum number of positions to convert. Defaults to number of scenes.
    """

    img = BioImage(dataset_config.original_path)

    if dataset_config.microscope == "3i":
        physical_pixel_sizes = get_sldy_pixel_sizes(img.metadata)
    if dataset_config.microscope == "Nikon":
        physical_pixel_sizes = img.physical_pixel_sizes

    interval_min = dataset_config.time_interval_in_minutes
    fmsid = dataset_config.fmsid
    n_positions = dataset_config.n_total_positions
    num_scenes = len(img.scenes)

    assert n_positions % num_scenes == 0, (
        f"Number of positions ({n_positions}) in dataset config must be divisible by "
        f"number of scenes ({num_scenes}) in the image file for dataset '{dataset_config.name}'"
    )

    num_pos_in_t = n_positions // num_scenes
    num_pos_in_s = num_scenes

    # Select which scenes to include in the converted file. By default,
    # include all available scenes.
    include_scenes = dataset_config.include_scenes
    if include_scenes is None:
        include_scenes = range(num_scenes)

    count = 0
    for scene_index in range(num_pos_in_s):
        # If current scene index is not in list of scenes to include, skip.
        if scene_index not in include_scenes:
            continue

        logger.info("Processing scene '%s'", img.scenes[scene_index])

        for position_index in range(num_pos_in_t):
            output = (
                f"{output_path}/{dataset_config.date}_{fmsid}/"
                f"{dataset_config.date}_{fmsid}_P{count}.ome.zarr"
            )
            print(f"Writing to {output}")
            scene = get_delayed_array_for_position(
                position_index, dataset_config, channel_names, num_pos_in_t, scene_index, img
            )
            write_scene(
                img=scene,
                full_zarr_path=output,
                image_name=f"{dataset_config.name}_{position_index}",
                channel_names=channel_names,
                max_timepoints=max_timepoints,
                physical_pixel_sizes=physical_pixel_sizes,
                interval_min=interval_min,
            )
            count += 1

            if count >= max_positions:
                print("Demo mode is ON. Processing only the first scene.")
                return
