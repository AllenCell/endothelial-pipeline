"""Methods for working with image manifests."""

import dataclasses
import logging
from pathlib import Path

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.manifests import ImageLocation, ImageManifest, load_image_manifest
from endo_pipeline.settings import ZARR_IMAGE_MANIFEST_NAME

logger = logging.getLogger(__name__)


def list_datasets_with_images(manifest: ImageManifest) -> list[str]:
    """Get list of dataset names that have valid image locations in the given manifest."""

    return [name for name, location in manifest.locations.items() if location.path is not None]


def get_image_location_for_dataset(
    manifest: ImageManifest,
    dataset: DatasetConfig,
    position: int | None = None,
    timepoint: int | None = None,
) -> ImageLocation:
    """Get the image location for the given dataset from the manifest, if it exists."""

    if dataset.name not in manifest.locations:
        logger.error(
            "Dataset [ %s ] does not have a location in image manifest [ %s ]",
            dataset.name,
            manifest.name,
        )
        raise KeyError(f"Unable to find dataset {dataset.name} in image manifest.")

    location = dataclasses.replace(manifest.locations[dataset.name])

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
            raise ValueError(f"Dataset {dataset.name} only has positions: {dataset.zarr_positions}")
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
            raise ValueError(f"Dataset {dataset.name} only has {dataset.duration} timepoints")
        elif "{{timepoint}}" not in str(location.path):
            logger.warning("Provided timepoint [ %d ] not used for location key", timepoint)
        else:
            location.path = Path(str(location.path).replace("{{timepoint}}", str(timepoint)))
    elif "{{timepoint}}" in str(location.path):
        logger.error("Dataset [ %s ] location requires timepoint", dataset.name)
        raise ValueError(f"Timepoint cannot be 'None' for location '{location.path}'")

    return location


def get_zarr_location_for_position(dataset: DatasetConfig, position: int) -> ImageLocation:
    """Get zarr image location for given dataset and position."""

    manifest = load_image_manifest(ZARR_IMAGE_MANIFEST_NAME)
    return get_image_location_for_dataset(manifest, dataset, position=position)


def get_available_zarr_locations(dataset: DatasetConfig) -> list[ImageLocation]:
    """Get list of all available Zarr locations for given dataset."""

    return [
        get_zarr_location_for_position(dataset, position)
        for position in sorted(dataset.zarr_positions)
    ]
