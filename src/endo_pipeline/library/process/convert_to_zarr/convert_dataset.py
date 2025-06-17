from bioio import BioImage

from cellsmap.util.dataset_io import (
    get_fmsid,
    get_microscope,
    get_original_path,
    get_time_interval_in_minutes,
    get_total_number_of_positions,
)
from src.endo_pipeline.library.process.convert_to_zarr.load_raw_image_data import (
    get_delayed_array_for_position,
    get_included_scenes,
)
from src.endo_pipeline.library.process.convert_to_zarr.write_zarr import (
    get_sldy_pixel_sizes,
    write_scene,
)


def convert_dataset(
    dataset: str,
    output_dataset_name: str,  # date
    output_path: str = "//allen/aics/endothelial/morphological_features/image_data/converted_zarrs",
    channel_names: list[str] = ["EGFP", "BF"],
) -> None:
    """
    Converts a raw dataset into a Zarr format with a specific channel order
    and images of positions over time are organized in scenes.

    Args:
        dataset (str): The name of the dataset to be converted.
        output_dataset_name (str): The name of the output dataset (typically includes a date).
        output_path (str, optional): The base directory where the converted Zarr files will be saved.
            Defaults to "//allen/aics/endothelial/morphological_features/image_data/converted_zarrs".
        channel_names (list[str], optional): A list of channel names to include in the output.
            Defaults to ["EGFP", "BF"].
    """

    img = BioImage(get_original_path(dataset))
    if get_microscope(dataset) == "3i":
        physical_pixel_sizes = get_sldy_pixel_sizes(img.metadata)
    if get_microscope(dataset) == "Nikon":
        physical_pixel_sizes = img.physical_pixel_sizes
    interval_min = get_time_interval_in_minutes(dataset)
    fmsid = get_fmsid(dataset)
    n_positions = get_total_number_of_positions(dataset)

    assert n_positions % len(img.scenes) == 0, (
        f"Number of positions ({n_positions}) in data_config.yaml must be divisible by "
        f"number of scenes ({len(img.scenes)}) in the image file for dataset {dataset}"
    )

    num_pos_in_T = n_positions // len(img.scenes)
    num_pos_in_S = len(img.scenes)

    count = 0
    for scene_index in range(num_pos_in_S):
        subset_scene_list = get_included_scenes(dataset)
        if scene_index not in subset_scene_list:
            continue
        else:
            print(f"Processing scene {img.scenes[scene_index]}")
        for position in range(num_pos_in_T):
            output = f"{output_path}/{output_dataset_name}_{fmsid}/{output_dataset_name}_{fmsid}_P{count}.ome.zarr"
            print(f"Writing to {output}")
            scene = get_delayed_array_for_position(
                position, dataset, channel_names, num_pos_in_T, scene_index, img
            )
            write_scene(
                scene,
                channel_names,
                output,
                dataset,
                position,
                physical_pixel_sizes,
                interval_min,
            )
            count += 1
