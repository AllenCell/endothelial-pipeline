import argparse

from src.endo_pipeline.library.process.convert_to_zarr.convert_dataset import (
    convert_dataset,
)

"""
This script processes images from a dataset and writes them to Zarr format.
Zarrs are saved in this default channel order: 488, BF, 405, 561, 640

Usage:
    python write_zarr.py <dataset> <output_path>

Arguments:
    dataset : str
        The name of the dataset to process.
    output_path : str
        The path where the Zarr files will be saved.

Example:

python cellsmap/image_conversion/raw_img_to_zarr.py 20250509_20X_IF3 20250509 --channel_names EGFP,BF,NucViolet,SOX17,SMAD1

Example (using API):
    output_path = Path('//allen/aics/assay-dev/users/Serge/test_images')
    convert_sldy_dataset(dataset='20240305_T01_001', output_path=output_path)

This will process the dataset '20240305_T01_001' and save the output to the specified directory.
The resulting zarr contains images from one scene.
"""


def parse_arguments() -> tuple[str, str, str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Process sldy or nd2 images and write to Zarr format."
    )
    parser.add_argument(
        "dataset", type=str, help="The dataset name matching dataset_config.yaml"
    )
    parser.add_argument(
        "output_dataset_name",
        type=str,
        help="The output dataset name for the Zarr files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="//allen/aics/endothelial/morphological_features/image_data/converted_zarrs",
        help="The output path for the Zarr files",
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


if __name__ == "__main__":
    dataset, output_dataset_name, output_path, channel_names = parse_arguments()
    convert_dataset(dataset, output_dataset_name, output_path, channel_names)
