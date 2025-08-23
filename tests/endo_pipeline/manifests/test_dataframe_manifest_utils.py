import pytest

from endo_pipeline.manifests.dataframe_manifest import DataframeLocation, DataframeManifest
from endo_pipeline.manifests.dataframe_manifest_utils import (
    get_dataframe_location_for_dataset,
    list_datasets_with_dataframes,
)


@pytest.fixture
def manifest():
    return DataframeManifest(
        name="unique_dataframe_manifest_name",
        workflow="workflow_name",
        locations={},
    )


def test_list_datasets_with_dataframes_with_valid_locations(manifest):
    manifest.locations = {
        "dataset_one": DataframeLocation(fmsid="dataset_one_fmsid"),
        "dataset_two": DataframeLocation(fmsid="dataset_two_fmsid", s3uri="s3://dataset_two"),
        "dataset_three": DataframeLocation(s3uri="s3://dataset_three"),
    }

    datasets = list_datasets_with_dataframes(manifest)

    assert datasets == ["dataset_one", "dataset_two", "dataset_three"]


def test_list_datasets_with_dataframes_with_invalid_location(manifest):
    manifest.locations = {
        "dataset_one": DataframeLocation(fmsid="dataset_one_fmsid"),
        "dataset_two": DataframeLocation(),
        "dataset_three": DataframeLocation(s3uri="s3://dataset_three"),
    }

    datasets = list_datasets_with_dataframes(manifest)

    assert datasets == ["dataset_one", "dataset_three"]


@pytest.mark.parametrize(
    "dataset_name,expected_fmsid,expected_s3uri",
    [
        ("dataset_one", "dataset_one_fmsid", None),
        ("dataset_two", "dataset_two_fmsid", "s3://dataset_two"),
        ("dataset_three", None, "s3://dataset_three"),
    ],
)
def test_get_dataframe_location_for_dataset_valid_dataset(
    manifest, dataset_name, expected_fmsid, expected_s3uri
):
    manifest.locations = {
        "dataset_one": DataframeLocation(fmsid="dataset_one_fmsid"),
        "dataset_two": DataframeLocation(fmsid="dataset_two_fmsid", s3uri="s3://dataset_two"),
        "dataset_three": DataframeLocation(s3uri="s3://dataset_three"),
    }

    location = get_dataframe_location_for_dataset(manifest, dataset_name)

    if expected_fmsid is None:
        assert location.fmsid is None
    else:
        assert location.fmsid == expected_fmsid

    if expected_s3uri is None:
        assert location.s3uri is None
    else:
        assert location.s3uri == expected_s3uri


def test_get_dataframe_location_for_dataset_invalid_dataset(manifest):
    with pytest.raises(KeyError):
        get_dataframe_location_for_dataset(manifest, "invalid_dataset")
