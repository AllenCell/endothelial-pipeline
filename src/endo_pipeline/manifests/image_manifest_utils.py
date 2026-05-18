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

    return [
        name
        for name, location in manifest.locations.items()
        if location.path is not None or location.s3uri is not None
    ]


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

    # If location has no path or uri defined, do not need to do position or
    # timepoint replacements so return as is.
    if location.path is None and location.s3uri is None:
        return location

    # If location does not have any placeholders, do not need to do position or
    # timepoint replacements so return as is. If a replacement position or
    # timepoint was provided, also raise a warning.
    placeholders = ("{{position}}", "{{timepoint}}")
    if all(ph not in str(location.path) and ph not in str(location.s3uri) for ph in placeholders):
        if position is not None:
            logger.warning("Position [ %d ] not used by location key", position)
        if timepoint is not None:
            logger.warning("Timepoint [ %d ] not used by location key", timepoint)
        return location

    # Check if the position is valid before making any replacements
    if position is not None and position not in dataset.zarr_positions:
        logger.warning("Position [ %d ] not valid for dataset [ %s ]", position, dataset.name)
        position = None

    # If there is a path with a placeholder, make the replacement
    if position is not None and location.path and "{{position}}" in str(location.path):
        location.path = Path(str(location.path).replace("{{position}}", str(position)))

    # If there is a uri with a placeholder, make the replacement
    if position is not None and location.s3uri and "{{position}}" in str(location.s3uri):
        location.s3uri = str(location.s3uri).replace("{{position}}", str(position))

    # Raise an expection if position is not provided but placeholder exists in path
    if "{{position}}" in str(location.path):
        logger.error("Dataset [ %s ] location requires position", dataset.name)
        raise ValueError(f"Position cannot be 'None' for location '{location.path}'")

    # Raise an expection if position is not provided but placeholder exists in uri
    if "{{position}}" in str(location.s3uri):
        logger.error("Dataset [ %s ] location requires position", dataset.name)
        raise ValueError(f"Position cannot be 'None' for location '{location.s3uri}'")

    # Check if the timepoint is valid before making any replacements
    if timepoint is not None and (timepoint < 0 or timepoint >= dataset.duration):
        logger.warning("Timepoint [ %d ] not valid for dataset [ %s ]", timepoint, dataset.name)
        timepoint = None

    # If there is a path with a placeholder, make the replacement
    if timepoint is not None and location.path and "{{timepoint}}" in str(location.path):
        location.path = Path(str(location.path).replace("{{timepoint}}", str(timepoint)))

    # If there is a uri with a placeholder, make the replacement
    if timepoint is not None and location.s3uri and "{{timepoint}}" in str(location.s3uri):
        location.s3uri = str(location.s3uri).replace("{{timepoint}}", str(timepoint))

    # Raise an expection if timepoint is not provided but placeholder exists in path
    if "{{timepoint}}" in str(location.path):
        logger.error("Dataset [ %s ] location requires timepoint", dataset.name)
        raise ValueError(f"Timepoint cannot be 'None' for location '{location.path}'")

    # Raise an expection if timepoint is not provided but placeholder exists in uri
    if "{{timepoint}}" in str(location.s3uri):
        logger.error("Dataset [ %s ] location requires timepoint", dataset.name)
        raise ValueError(f"Timepoint cannot be 'None' for location '{location.s3uri}'")

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


def add_image_location_to_manifest(
    manifest: ImageManifest,
    dataset: DatasetConfig,
    parent_path: str | Path,
) -> None:
    """
    Add or update image location for given dataset in the manifest.

    The path is formatted as:

        PARENT_PATH / DATE_FMSID / DATE_FMSID_P{{position}}.ome.zarr

    Parameters
    ----------
    manifest
        Image manifest object.
    dataset
        Dataset config object.
    parent_path
        Path to the parent location for the image.
    """

    date = dataset.date
    fmsid = dataset.fmsid
    path = Path(parent_path) / f"{date}_{fmsid}" / f"{date}_{fmsid}_P{{{{position}}}}.ome.zarr"

    if dataset.name in manifest.locations.keys():
        logger.warning("Dataset [ %s ] has existing location and will be overwritten", dataset.name)

    manifest.locations[dataset.name] = ImageLocation(path=path)


def build_image_location_from_string(string: str) -> ImageLocation:
    """Create a image location from given string."""

    if string.startswith("s3://"):
        return ImageLocation(s3uri=string)
    else:
        return ImageLocation(path=Path(string).resolve())
