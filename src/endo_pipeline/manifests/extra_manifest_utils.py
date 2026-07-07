"""Methods for working with extra manifests."""

import dataclasses
import logging
from pathlib import Path

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.manifests import ExtraLocation, ExtraManifest

logger = logging.getLogger(__name__)


def get_extra_location_for_dataset(
    manifest: ExtraManifest,
    dataset: DatasetConfig,
    position: int | None = None,
) -> ExtraLocation:
    """Get the extra location for the given dataset from the manifest, if it exists."""

    if dataset.name not in manifest.locations:
        logger.error(
            "Dataset [ %s ] does not have a location in extra manifest [ %s ]",
            dataset.name,
            manifest.name,
        )
        raise KeyError(f"Unable to find dataset {dataset.name} in extra manifest.")

    location = dataclasses.replace(manifest.locations[dataset.name])

    # If location has no path or uri defined, do not need to do position
    # replacements so return as is.
    if location.path is None and location.s3uri is None:
        return location

    # If location does not have any placeholders, do not need to do position
    # replacements so return as is. If a replacement position was provided, also
    # raise a warning.
    placeholders = ("{{position}}",)
    if all(ph not in str(location.path) and ph not in str(location.s3uri) for ph in placeholders):
        if position is not None:
            logger.warning("Position [ %d ] not used by location key", position)
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

    return location
