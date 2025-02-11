import argparse
from cellsmap.util.io import get_number_of_positions
from cellsmap.image_conversion.process_images.process_sldy import process_position
from cellsmap.image_conversion.process_images.write_zarr import (
    write_scene,
    get_sldy_metadata,
)

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
    python write_zarr.py 20240305_T01_001 /allen/aics/assay-dev/users/Chantelle/outputs/temp_tiffs

This will process the dataset '20240305_T01_001' and save the output to the specified directory.
"""

CHANNEL_NAMES = ["egfp", "bf"]


def convert_sldy_dataset(dataset: str, output_path: str):
    n_positions = get_number_of_positions(dataset)
    physical_pixel_sizes = get_sldy_metadata(dataset)
    for position in range(n_positions):
        output = f"{output_path}/{dataset}/{dataset}_{position}.ome.zarr"
        scene = process_position(position, dataset)
        write_scene(
            scene, CHANNEL_NAMES, output, dataset, position, physical_pixel_sizes
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

    args = parser.parse_args()

    convert_sldy_dataset(args.dataset, args.output_path)


if __name__ == "__main__":
    main()
