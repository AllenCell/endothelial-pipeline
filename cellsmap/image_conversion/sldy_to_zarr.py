# %%
import argparse
from cellsmap.util.dataset_io import (
    get_total_number_of_positions, 
    get_time_interval_in_minutes,
    get_microscope,
    get_fmsid,)
from cellsmap.image_conversion.process_images.process_sldy import get_delayed_array_for_position
from cellsmap.image_conversion.process_images.write_zarr import (
    write_scene,
    get_sldy_pixel_sizes,
)
from cellsmap.util.dataset_io import get_original_path
from bioio import BioImage
from pathlib import Path

# %%
"""
This script processes images from a dataset and writes them to Zarr format.

Usage:
    python write_zarr.py <dataset> <output_path>

Arguments:
    dataset : str
        The name of the dataset to process.
    output_path : str
        The path where the Zarr files will be saved.

Example:
    python cellsmap/image_conversion/sldy_to_zarr.py 20241016_20X 20241016 /allen/aics/endothelial/morphological_features/image_data/converted_zarrs

Example (using API):
    output_path = Path('//allen/aics/assay-dev/users/Serge/test_images')
    convert_sldy_dataset(dataset='20240305_T01_001', output_path=output_path)

This will process the dataset '20240305_T01_001' and save the output to the specified directory.
The resulting zarr contains images from one scene.
"""


def convert_sldy_dataset(
    dataset: str,
    output_path: str,
    output_dataset_name: str,  # barcode_date
    channel_names: list[str] = ["EGFP", "BF"],
):
    img = BioImage(get_original_path(dataset))
    if get_microscope(dataset) == "3i":
        physical_pixel_sizes = get_sldy_pixel_sizes(img.metadata)
    if get_microscope(dataset) == "Nikon":
        physical_pixel_sizes = img.physical_pixel_sizes
    interval_min = get_time_interval_in_minutes(dataset)
    fmsid = get_fmsid(dataset)
    n_positions = get_total_number_of_positions(dataset)

    assert n_positions % len(img.scenes) == 0, \
        f'Number of positions ({n_positions}) in data_config.yaml must be divisible by ' \
        f'number of scenes ({len(img.scenes)}) in the image file for dataset {dataset}'
    
    num_pos_in_T = n_positions // len(img.scenes)
    num_pos_in_S = len(img.scenes)
    
    count = 0
    for scene_index in range(num_pos_in_S):
        for position in range(num_pos_in_T):
            output = f"{output_path}/{output_dataset_name}_{fmsid}/{output_dataset_name}_{fmsid}_P{count}.ome.zarr"
            print(f"Writing to {output}")
            scene = get_delayed_array_for_position(
                position, dataset, num_pos_in_T, scene_index, img
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

def main(dataset: str, output_path: str, output_dataset_name: str, channel_names: list):
    convert_sldy_dataset(dataset, output_path, output_dataset_name, channel_names)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process sldy images and write to Zarr format."
    )
    parser.add_argument(
        "dataset", type=str, help="The dataset name matching dataset_config.yaml"
    )
    parser.add_argument(
        "output_dataset_name",
        type=str,
        help="The output datset name for the Zarr files",
    )
    parser.add_argument(
        "output_path", 
        type=str, 
        help="The output path for the Zarr files"
    )
    parser.add_argument(
        "--channel_names",
        type=str,
        default="EGFP,BF",
        help="Comma-separated list of channel names",
    )

    args = parser.parse_args()
    channel_names = args.channel_names.split(",")
    return args.dataset, args.output_dataset_name, args.output_path, channel_names


# %%
if __name__ == "__main__":
    dataset, output_dataset_name, output_path, channel_names = parse_arguments()
    main(dataset, output_path, output_dataset_name, channel_names)
# %%
