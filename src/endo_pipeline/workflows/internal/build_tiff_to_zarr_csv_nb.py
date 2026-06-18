"""
Create Tiff to Zarr conversion CSV.

#internal #zarr-conversion

This notebook iterates through all datasets and positions for the following
segmentations and compiles a CSV containing paths to the original Zarr.

- nuclear label free segmentation (`nuclear_labelfree_seg`)
- CDH5 classic segmentation (`cdh5_classic_seg`)
- grid segmentation (`grid_seg`)
- CDH5 segmentation validation (`cdh5_seg_validations`)

The output CSV contains the following columns:

| Column Name      | Description                                         |
| ---------------- | --------------------------------------------------- |
| `dataset_name`   | Name of the dataset                                 |
| `original_zarr`  | Path to the original Zarr image                     |
| `tiff_seg_dir`   | Path to the input directory of Tiff segmentations   |
| `save_zarr_path` | Path to the output directory for Zarr segmentations |
| `duration`       | Total number of timepoints in the dataset           |
| `channel_name`   | Name of the channel                                 |
"""

# %%
import logging
from pathlib import Path

import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.manifests import (
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    list_datasets_with_images,
    load_image_manifest,
)

# %%
logger = logging.getLogger(__name__)

zarr_seg_dir = Path("//allen/aics/endothelial/morphological_features/segmentations/")
image_channel_pairs = [
    ("nuclear_labelfree_seg", "NUC_SEG"),
    ("cdh5_classic_seg", "CDH5_SEG"),
    ("grid_seg", "GRID_SEG"),
    (
        "cdh5_seg_validations",
        [
            "Raw CDH5",
            "Processed CDH5",
            "Hysteresis Threshold",
            "Initial Segmentation",
            "Merged Segmentation",
            "Nuclei Predictions",
            "CDH5 Segmentation Split by Nuclei",
            "CDH5 Segmentation Split by Nuclei Borders",
        ],
    ),
]

# %%
for manifest_name, channel_name in image_channel_pairs:
    logger.info("Processing image manifest '%s' channel '%s'", manifest_name, channel_name)

    zarr_seg_path = zarr_seg_dir / f"{manifest_name}_zarr"
    output_path = get_output_path("tiff_to_zarr")

    image_manifest = load_image_manifest(manifest_name)
    datasets = list_datasets_with_images(image_manifest)

    # Grid segmentations are the same for all datasets, so limit the dataset
    # list to just the first dataset for the following loop.
    if manifest_name == "grid_seg":
        datasets = datasets[:1]

    # Segmentation validations have multiple channels, so join into a string
    # that will later get split back into channels
    if manifest_name == "cdh5_seg_validations":
        channel_name = "/".join(channel_name)

    data = []

    for dataset_name in datasets:
        try:
            dataset_config = load_dataset_config(dataset_name)
        except Exception as e:
            logger.error("Failed to load dataset config for '%s': %s", dataset_name, e)
            continue

        for position in dataset_config.zarr_positions:
            # Get input path to Tiff segmentations
            location = get_image_location_for_dataset(
                image_manifest, dataset_config, position=position, timepoint=0
            )
            assert location.path is not None
            tiff_seg_dir = location.path.parent

            # Get path to the original Zarr file
            original_zarr_loc = get_zarr_location_for_position(dataset_config, position)
            assert original_zarr_loc.path is not None
            original_zarr_path = original_zarr_loc.path

            # Create output path for Zarr segmentations
            if manifest_name == "grid_seg":
                save_path = zarr_seg_path / f"P{position}.ome.zarr"
            else:
                save_path = zarr_seg_path / Path(*original_zarr_path.parts[-2:])

            # Check if the output path already exists
            if save_path.exists():
                logger.debug("Converted zarr '%s' already exists. Skipping.", save_path)
                continue

            # Get dataset duration  because Tiff segmentations are individual
            # timepoints for each file, but the converted Zarr segmentation will
            # have all timpoints in a single file.
            duration = dataset_config.duration

            data.append(
                {
                    "dataset_name": dataset_name,
                    "original_zarr": str(original_zarr_path),
                    "tiff_seg_dir": str(tiff_seg_dir),
                    "save_zarr_path": str(save_path),
                    "duration": duration,
                    "channel_name": channel_name,
                }
            )

    df = pd.DataFrame(data)
    df.to_csv(output_path / f"tiff_to_zarr_conversion_{manifest_name}.csv", index=False)

# %%
