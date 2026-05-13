from bioio import BioImage

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.library.process.convert_to_zarr.load_raw_image_data import (
    get_delayed_array_for_position,
    get_included_scenes,
)
from endo_pipeline.library.process.convert_to_zarr.write_zarr import (
    get_sldy_pixel_sizes,
    write_scene,
)


def convert_dataset(
    dataset_config: DatasetConfig,
    output_path: str,
    channel_names: list[str],
    demo_mode: bool = False,
) -> None:
    """
    Convert raw dataset to Zarr format with specific channel order.

    Parameters
    ----------
    dataset_config
        Dataset config for the dataset to be converted.
    output_path
        The base directory where the converted Zarr files will be saved.
    channel_names
        A list of channel names to include in the output.
    demo_mode
        If True, process only the first scene for testing purposes
    """

    img = BioImage(dataset_config.original_path)

    if dataset_config.microscope == "3i":
        physical_pixel_sizes = get_sldy_pixel_sizes(img.metadata)
    if dataset_config.microscope == "Nikon":
        physical_pixel_sizes = img.physical_pixel_sizes

    interval_min = dataset_config.time_interval_in_minutes
    fmsid = dataset_config.fmsid
    n_positions = dataset_config.n_total_positions
    max_timepoints = 10 if demo_mode else dataset_config.duration

    assert n_positions % len(img.scenes) == 0, (
        f"Number of positions ({n_positions}) in data_config.yaml must be divisible by "
        f"number of scenes ({len(img.scenes)}) in the image file for dataset {dataset_config.name}"
    )

    num_pos_in_t = n_positions // len(img.scenes)
    num_pos_in_s = len(img.scenes)

    count = 0
    for scene_index in range(num_pos_in_s):
        subset_scene_list = get_included_scenes(dataset_config)
        if scene_index not in subset_scene_list:
            continue
        else:
            print(f"Processing scene {img.scenes[scene_index]}")
        for position in range(num_pos_in_t):
            output = (
                f"{output_path}/{dataset_config.date}_{fmsid}/"
                f"{dataset_config.date}_{fmsid}_P{count}.ome.zarr"
            )
            print(f"Writing to {output}")
            scene = get_delayed_array_for_position(
                position, dataset_config, channel_names, num_pos_in_t, scene_index, img
            )
            write_scene(
                scene,
                channel_names,
                output,
                dataset_config.name,
                position,
                max_timepoints,
                physical_pixel_sizes,
                interval_min,
            )
            count += 1

            if demo_mode and count > 0:
                print("Demo mode is ON. Processing only the first scene.")
                return
