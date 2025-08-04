from pathlib import Path

import pytest

from src.endo_pipeline.manifests.segmentation_manifest import (
    SegmentationLocation,
    SegmentationManifest,
)
from src.endo_pipeline.manifests.segmentation_manifest_utils import (
    get_segmentation_location_for_dataset,
    list_datasets_with_segmentations,
)


@pytest.fixture
def manifest():
    return SegmentationManifest(
        name="unique_segmentation_manifest_name",
        workflow="workflow_name",
        locations={
            "dataset_one": SegmentationLocation(path=Path("/path/to/dataset_one/seg.ome.tiff")),
            "dataset_two": SegmentationLocation(path=Path("/path/to/dataset_two/seg.ome.tiff")),
            "dataset_three": SegmentationLocation(path=Path("/path/to/dataset_three/seg.ome.tiff")),
        },
    )


def test_list_datasets_with_segmentations_with_valid_locations(manifest):
    datasets = list_datasets_with_segmentations(manifest)

    assert datasets == ["dataset_one", "dataset_two", "dataset_three"]


def test_list_datasets_with_segmentations_with_invalid_location(manifest):
    manifest.locations["dataset_two"] = SegmentationLocation()

    datasets = list_datasets_with_segmentations(manifest)

    assert datasets == ["dataset_one", "dataset_three"]


@pytest.mark.parametrize(
    "dataset_name,expected_path",
    [
        ("dataset_one", "/path/to/dataset_one/seg.ome.tiff"),
        ("dataset_two", "/path/to/dataset_two/seg.ome.tiff"),
        ("dataset_three", "/path/to/dataset_three/seg.ome.tiff"),
    ],
)
def test_get_segmentation_location_for_dataset_valid_dataset(manifest, dataset_name, expected_path):
    location = get_segmentation_location_for_dataset(manifest, dataset_name)

    assert location.path.as_posix() == expected_path


def test_get_segmentation_location_for_dataset_invalid_dataset(manifest):
    with pytest.raises(KeyError):
        get_segmentation_location_for_dataset(manifest, "invalid_dataset")
