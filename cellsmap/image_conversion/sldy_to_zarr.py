import argparse
from cellsmap.util.io import get_number_of_positions, get_time_interval_in_minutes
from cellsmap.image_conversion.process_images.process_sldy import get_delayed_array_for_position
from cellsmap.image_conversion.process_images.write_zarr import (
    write_scene,
    get_sldy_metadata,
)
from cellsmap.util.io import get_original_path
from bioio import BioImage
from pathlib import Path

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

Example (using API):
    output_path = Path('/allen/aics/assay-dev/users/Serge/test_images')
    convert_sldy_dataset(dataset=20240305_T01_001, output_path=str(output_path))

This will process the dataset '20240305_T01_001' and save the output to the specified directory.
The resulting zarr contains images from one scene.
"""

def convert_sldy_dataset(dataset: str, output_path: str, channel_names: list[str] = ["EGFP", "BF"]):
    # NOTE there is an implicit assumption here that all scenes in a dataset
    #       have the same number of positions, which may not always be true.
    n_positions = get_number_of_positions(dataset)
    physical_pixel_sizes = get_sldy_metadata(dataset)
    interval_min = get_time_interval_in_minutes(dataset)
    img = BioImage(get_original_path(dataset))
    assert not (n_positions > 1 and len(img.scenes) > 1), "One of number of positions or number of scenes must be one."
    for scene_index in range(len(img.scenes)):
        for position in range(n_positions):
            if n_positions > 1:
                output = f"{output_path}/{dataset}/{dataset}_P{position}.ome.zarr"
            else:
                output = f"{output_path}/{dataset}/{dataset}_P{scene_index}.ome.zarr"
            print(f"Writing to {output}")
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
    main(dataset='20240305_T01_001', output_path=Path("/allen/aics/assay-dev/users/Serge/test_images"))
