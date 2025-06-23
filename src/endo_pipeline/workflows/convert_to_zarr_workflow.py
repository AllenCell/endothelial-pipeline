import argparse

from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.process.convert_to_zarr.convert_dataset import (
    convert_dataset,
)

"""
This script processes images from a dataset and writes them to Zarr format.
Zarrs are saved in this default channel order: 488, BF, 405, 561, 640.

Usage:
    python write_zarr.py <dataset> <output_path>

Arguments:
    dataset : str
        The name of the dataset to process.
    output_path : str
        The path where the Zarr files will be saved. Default is the results folder.
    channel_names : str
        Comma-separated list of channel names to include in the Zarr files.
        Default is "EGFP,BF".
        For IF data, name the marker imaged by that channel, e.g.,
        "EGFP,BF,NucViolet,SOX17,SMAD1".

Example to test:
    python src/endo_pipeline/workflows/convert_to_zarr_workflow.py 20250509_20X_IF3 \
        20250509 --channel_names EGFP,BF,NucViolet,SOX17,SMAD1

Example to run:
    python src/endo_pipeline/workflows/convert_to_zarr_workflow.py 20250509_20X_IF3 \
        20250509 --channel_names EGFP,BF,NucViolet,SOX17,SMAD1 \
        --output_path \
        //allen/aics/endothelial/morphological_features/image_data/converted_zarrs
"""


def parse_arguments() -> tuple[str, str, str, list[str]]:
    """Parse command-line arguments for the Zarr conversion workflow."""
    parser = argparse.ArgumentParser(
        description="Process sldy or nd2 images and write to Zarr format."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="The dataset name matching dataset_config.yaml",
    )
    parser.add_argument(
        "output_dataset_name",
        type=str,
        help="The output dataset name for the Zarr files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=get_output_path("zarr_conversion"),
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
