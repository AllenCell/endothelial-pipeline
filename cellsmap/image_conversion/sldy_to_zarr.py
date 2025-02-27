import argparse
from cellsmap.util.io import get_number_of_positions, get_time_interval_in_minutes
from cellsmap.image_conversion.process_images.process_sldy import get_delayed_array_for_position
from cellsmap.image_conversion.process_images.write_zarr import (
    write_scene,
    get_sldy_metadata,
)
from cellsmap.util.io import get_original_path
from bioio import BioImage

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
    python cellsmap/image_conversion/sldy_to_zarr.py 20240305_T01_001 /allen/aics/assay-dev/users/Chantelle/outputs/temp

This will process the dataset '20240305_T01_001' and save the output to the specified directory.
"""

def convert_sldy_dataset(dataset: str, output_path: str, channel_names: list[str] = ["EGFP", "BF"]):
    # NOTE there is an implicit assumption here that all scenes in a dataset
    #       have the same number of positions, which may not always be true.
    # TODO handle the case where the number of positions varies between scenes
    n_positions = get_number_of_positions(dataset)
    physical_pixel_sizes = get_sldy_metadata(dataset)
    interval_min = get_time_interval_in_minutes(dataset)
    img = BioImage(get_original_path(dataset))
    for scene_index in range(len(img.scenes)):
        for position in range(n_positions):
            output = f"{output_path}/{dataset}/S{scene_index}/{dataset}_S{scene_index}_P{position}.ome.zarr"
            scene = get_delayed_array_for_position(position, dataset, n_positions, scene_index)
            write_scene(
                scene, channel_names, output, dataset, position, physical_pixel_sizes, interval_min
            )


def main():
    parser = argparse.ArgumentParser(
        description="Process sldy images and write to Zarr format."
    )
    parser.add_argument(
        "dataset", type=str, help="The dataset name matching dataset_config.yaml"
    )
    parser.add_argument(
        "output_path", type=str, help="The output path for the Zarr files"
    )
    parser.add_argument(
        "--channel_names", type=str, default="EGFP,BF", help="Comma-separated list of channel names"
    )

    args = parser.parse_args()
    channel_names = args.channel_names.split(',')

    convert_sldy_dataset(args.dataset, args.output_path, channel_names)


if __name__ == "__main__":
    main()
