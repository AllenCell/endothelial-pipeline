"""Methods for working with segmentation manifests."""

import dataclasses
import logging
from pathlib import Path

from src.endo_pipeline.configs import load_dataset_config
from src.endo_pipeline.manifests import ImageLocation, ImageManifest

logger = logging.getLogger(__name__)


def list_datasets_with_segmentations(manifest: ImageManifest) -> list[str]:
    """Get list of dataset names that have valid segmentation locations in the given manifest."""

    return [name for name, location in manifest.locations.items() if location.path is not None]


def get_segmentation_location_for_dataset(
    manifest: ImageManifest,
    dataset_name: str,
    position: int | None = None,
    timepoint: int | None = None,
) -> ImageLocation:
    """Get the segmentation location for the given dataset from the manifest, if it exists."""

    if dataset_name not in manifest.locations:
        logger.error(
            "Dataset [ %s ] does not have a location in segmentation manifest [ %s ]",
            dataset_name,
            manifest.name,
        )
        raise KeyError(f"Unable to find dataset {dataset_name} in segmentation manifest.")

    dataset = load_dataset_config(dataset_name)
    location = dataclasses.replace(manifest.locations[dataset_name])

    # If location has no path defined, do not need to do position or timepoint
    # replacements so return as is.
    if location.path is None:
        return location

    # If position is provided, replace any uses of {{position}} with the given
    # position (if valid for the dataset). If position is not provided and
    # {{position}} is in the location, raise an exception.
    if position is not None:
        if position not in dataset.zarr_positions:
            logger.error("Position [ %d ] not valid for dataset [ %s ]", position, dataset.name)
            raise ValueError(f"Dataset {dataset_name} only has positions: {dataset.zarr_positions}")
        elif "{{position}}" not in str(location.path):
            logger.warning("Provided position [ %d ] not used for location key", position)
        else:
            location.path = Path(str(location.path).replace("{{position}}", str(position)))
    elif "{{position}}" in str(location.path):
        logger.error("Dataset [ %s ] location requires position", dataset.name)
        raise ValueError(f"Position cannot be 'None' for location '{location.path}'")

    # If timepoint is provided, replace any uses of {{timepoint}} with the given
    # timepoint (if valid for the dataset). If timepoint is not provided and
    # {{timepoint}} is in the location, raise an exception.
    if timepoint is not None:
        if timepoint < 0 or timepoint >= dataset.duration:
            logger.error("Timepoint [ %d ] not valid for dataset [ %s ]", timepoint, dataset.name)
            raise ValueError(f"Dataset {dataset_name} only has {dataset.duration} timepoints")
        elif "{{timepoint}}" not in str(location.path):
            logger.warning("Provided timepoint [ %d ] not used for location key", timepoint)
        else:
            location.path = Path(str(location.path).replace("{{timepoint}}", str(timepoint)))
    elif "{{timepoint}}" in str(location.path):
        logger.error("Dataset [ %s ] location requires timepoint", dataset.name)
        raise ValueError(f"Timepoint cannot be 'None' for location '{location.path}'")

    return location
