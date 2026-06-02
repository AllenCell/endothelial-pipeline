"""Methods for working with dataframe manifests."""

import logging
from pathlib import Path

from endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    get_dataframe_manifest_dir,
    load_dataframe_manifest,
)

logger = logging.getLogger(__name__)


def list_datasets_with_dataframes(manifest: DataframeManifest) -> list[str]:
    """Get list of dataset names that have valid dataframe locations in the given manifest."""

    return [
        name
        for name, location in manifest.locations.items()
        if location.fmsid is not None or location.s3uri is not None or location.path is not None
    ]


def get_dataframe_location_for_dataset(
    manifest: DataframeManifest, dataset_name: str
) -> DataframeLocation:
    """Get the dataframe location for the given dataset from the manifest, if it exists."""

    if dataset_name not in manifest.locations:
        logger.error(
            "Dataset [ %s ] does not have a location in dataframe manifest [ %s ]",
            dataset_name,
            manifest.name,
        )
        raise KeyError(f"Unable to find dataset {dataset_name} in dataframe manifest.")

    return manifest.locations[dataset_name]


def get_dataframe_manifest_with_parameters(
    workflow: str, parameters: dict | None = None
) -> DataframeManifest:
    """Load dataframe manifest with matching workflow containing given parameters."""

    manifests = []
    parameters = parameters or {}

    # Iterate through all dataframe manifests to find ones with matching
    # workflow name and containing all given parameters.
    for manifest_file in get_dataframe_manifest_dir().iterdir():
        manifest = load_dataframe_manifest(manifest_file.stem)

        if manifest.workflow != workflow:
            continue

        if parameters.items() <= manifest.parameters.items():
            manifests.append(manifest)

    # If no manifests are found, raise error.
    if len(manifests) == 0:
        logger.error(
            "No dataframe manifests found for workflow '%s' with parameters %s",
            workflow,
            parameters,
        )
        raise LookupError("Unable to find manifest with matching parameters")

    # If multiple manifests are found, then raise error. We could also instead
    # raise a warning and return the first manifest found.
    if len(manifests) > 1:
        logger.error(
            "Multiple dataframe manifests found with given parameters: %s",
            " | ".join([manifest.name for manifest in manifests]),
        )
        raise ValueError("Found multiple manifests with matching parameters")

    return manifests[0]


def build_dataframe_location_from_path(path: str | Path) -> DataframeLocation:
    """Create a dataframe location from path."""

    # TODO: remove in favor of build_dataframe_location_from_string OR
    # directly creating the DataframeLocation object
    return DataframeLocation(path=Path(path).resolve())


def build_dataframe_location_from_string(string: str) -> DataframeLocation:
    """Create a dataframe location from given string."""

    if string.startswith("s3://"):
        return DataframeLocation(s3uri=string)
    else:
        return DataframeLocation(path=Path(string).resolve())
