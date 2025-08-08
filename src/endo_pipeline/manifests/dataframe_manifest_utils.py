"""Methods for working with dataframe manifests."""

import logging

from src.endo_pipeline.manifests import DataframeLocation, DataframeManifest

logger = logging.getLogger(__name__)


def list_datasets_with_dataframes(manifest: DataframeManifest) -> list[str]:
    """Get list of dataset names that have valid dataframe locations in the given manifest."""

    return [
        name
        for name, location in manifest.locations.items()
        if location.fmsid is not None or location.s3uri is not None
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
