import json
from contextlib import nullcontext

import pytest

from endo_pipeline.manifests.dataframe_manifest import DataframeLocation, DataframeManifest
from endo_pipeline.manifests.dataframe_manifest_io import get_dataframe_manifest_dir
from endo_pipeline.manifests.dataframe_manifest_utils import (
    get_dataframe_location_for_dataset,
    get_dataframe_manifest_with_parameters,
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


@pytest.mark.parametrize(
    "workflow,parameters,expected",
    [
        ("workflow_a", None, pytest.raises(ValueError)),
        ("workflow_b", None, nullcontext("B")),
        ("workflow_c", None, pytest.raises(LookupError)),
        ("workflow_a", {}, pytest.raises(ValueError)),
        ("workflow_b", {}, nullcontext("B")),
        ("workflow_a", {"param1": "A"}, nullcontext("A")),
        ("workflow_a", {"param1": "B"}, nullcontext("C")),
        ("workflow_a", {"param2": 2}, pytest.raises(ValueError)),
        ("workflow_a", {"param2": 3}, pytest.raises(LookupError)),
        ("workflow_a", {"param1": "A", "param2": 2}, nullcontext("A")),
        ("workflow_a", {"param1": "B", "param2": 2}, nullcontext("C")),
        ("workflow_b", {"param1": "B"}, nullcontext("B")),
    ],
)
def test_get_dataframe_manifest_with_parameters(fs, workflow, parameters, expected):
    manifest_dir = get_dataframe_manifest_dir()

    manifests = {
        "A": {"name": "A", "workflow": "workflow_a", "parameters": {"param1": "A", "param2": 2}},
        "B": {"name": "B", "workflow": "workflow_b", "parameters": {"param1": "B"}},
        "C": {"name": "C", "workflow": "workflow_a", "parameters": {"param1": "B", "param2": 2}},
    }

    fs.create_file(manifest_dir / "a.yaml", contents=json.dumps(manifests["A"]))
    fs.create_file(manifest_dir / "b.yaml", contents=json.dumps(manifests["B"]))
    fs.create_file(manifest_dir / "c.yaml", contents=json.dumps(manifests["C"]))

    with expected as e:
        manifest = get_dataframe_manifest_with_parameters(workflow, parameters)
        assert manifest == DataframeManifest(**manifests[e])
