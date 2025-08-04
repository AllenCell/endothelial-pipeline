"""Methods for working with segmentation manifests."""

import logging

from src.endo_pipeline.manifests import SegmentationLocation, SegmentationManifest

logger = logging.getLogger(__name__)


def list_datasets_with_segmentations(manifest: SegmentationManifest) -> list[str]:
    """Get list of dataset names that have valid segmentation locations in the given manifest."""

    return [name for name, location in manifest.locations.items() if location.path is not None]


def get_segmentation_location_for_dataset(
    manifest: SegmentationManifest, dataset_name: str
) -> SegmentationLocation:
    """Get the segmentation location for the given dataset from the manifest, if it exists."""

    if dataset_name not in manifest.locations:
        logger.error(
            "Dataset [ %s ] does not have a location in segmentation manifest [ %s ]",
            dataset_name,
            manifest.name,
        )
        raise KeyError("Unable to find dataset %s in segmentation manifest.")

    return manifest.locations[dataset_name]
