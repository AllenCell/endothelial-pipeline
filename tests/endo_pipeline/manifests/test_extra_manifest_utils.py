from pathlib import Path

import pytest

from endo_pipeline.configs import ChannelIndices, DatasetConfig
from endo_pipeline.manifests.extra_manifest import ExtraLocation, ExtraManifest
from endo_pipeline.manifests.extra_manifest_utils import get_extra_location_for_dataset


@pytest.fixture
def manifest():
    return ExtraManifest(
        name="unique_extra_manifest_name",
        workflow="workflow_name",
        locations={},
    )


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        name="",
        date="",
        original_path="",
        zarr_positions=[1, 3, 5],
        fmsid="",
        barcode="",
        cell_lines=[""],
        live_or_fixed_sample="live",
        is_timelapse=True,
        microscope="3i",
        objective="20X",
        shear_stress_regime=[],
        pixel_size_xy_in_um=0.0,
        duration=0,
        time_interval_in_minutes=0.0,
        channel_names=[],
        flow_conditions=[],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
    )


@pytest.fixture
def mock_load_extra_manifest(mocker):
    def _mocker(manifest_name, extra_manifest):
        manifest_mock = mocker.patch(
            "endo_pipeline.manifests.extra_manifest_utils.load_extra_manifest"
        )
        manifest_mock.side_effect = lambda x: extra_manifest if x == manifest_name else None

    return _mocker


@pytest.fixture
def expected_path(position, manifest_path):
    if manifest_path is None:
        return None
    elif position is None and "{{position}}" in str(manifest_path):
        return "invalid"
    elif (position == 4 or position == -1) and "{{position}}" in str(manifest_path):
        return "invalid"
    else:
        expected_path = manifest_path.replace("{{position}}", str(position))
        return Path(expected_path)


@pytest.fixture
def expected_s3uri(position, manifest_s3uri):
    if manifest_s3uri is None:
        return None
    elif position is None and "{{position}}" in manifest_s3uri:
        return "invalid"
    elif (position == 4 or position == -1) and "{{position}}" in manifest_s3uri:
        return "invalid"
    else:
        expected_s3uri = manifest_s3uri.replace("{{position}}", str(position))
        return expected_s3uri


@pytest.mark.parametrize(
    "position",
    [None, 3, 4, -1],
    ids=["no_position", "valid_position", "invalid_position", "negative_position"],
)
@pytest.mark.parametrize(
    "manifest_path",
    [
        None,
        "/path/to/seg/seg.ome.tiff",
        "/path/to/seg/P{{position}}_seg.ome.tiff",
    ],
    ids=lambda path: (
        "no_path"
        if path is None
        else "_".join(
            [
                "path",
                "with" if "{{position}}" in path else "without",
                "position_placeholder",
            ]
        )
    ),
)
@pytest.mark.parametrize(
    "manifest_s3uri",
    [
        None,
        "s3://bucket-name/seg.ome.tiff",
        "s3://bucket-name/P{{position}}_seg.ome.tiff",
    ],
    ids=lambda s3uri: (
        "no_s3uri"
        if s3uri is None
        else "_".join(
            [
                "s3uri",
                "with" if "{{position}}" in s3uri else "without",
                "position_placeholder",
            ]
        )
    ),
)
def test_get_extra_location_for_dataset_valid_dataset(
    dataset_config,
    manifest,
    manifest_path,
    manifest_s3uri,
    position,
    expected_path,
    expected_s3uri,
):
    if expected_path == "invalid" or expected_s3uri == "invalid":
        pytest.skip("Skipping invalid parameter combination for testing valid combinations")

    dataset_config.name = "dataset_name"
    dataset_config.zarr_positions = [1, 3, 5]
    dataset_config.duration = 10

    manifest.locations["dataset_name"] = ExtraLocation(path=manifest_path, s3uri=manifest_s3uri)

    location = get_extra_location_for_dataset(manifest, dataset_config, position=position)

    if manifest_path is None:
        assert location.path is None
    else:
        assert location.path == expected_path

    if manifest_s3uri is None:
        assert location.s3uri is None
    else:
        assert location.s3uri == expected_s3uri


@pytest.mark.parametrize(
    "position",
    [None, 3, 4, -1],
    ids=["no_position", "valid_position", "invalid_position", "negative_position"],
)
@pytest.mark.parametrize(
    "manifest_path",
    [
        None,
        "/path/to/seg/seg.ome.tiff",
        "/path/to/seg/P{{position}}_seg.ome.tiff",
    ],
    ids=lambda path: (
        "no_path"
        if path is None
        else "_".join(
            [
                "path",
                "with" if "{{position}}" in path else "without",
                "position_placeholder",
            ]
        )
    ),
)
@pytest.mark.parametrize(
    "manifest_s3uri",
    [
        None,
        "s3://bucket-name/seg.ome.tiff",
        "s3://bucket-name/P{{position}}_seg.ome.tiff",
    ],
    ids=lambda s3uri: (
        "no_s3uri"
        if s3uri is None
        else "_".join(
            [
                "s3uri",
                "with" if "{{position}}" in s3uri else "without",
                "position_placeholder",
            ]
        )
    ),
)
def test_get_extra_location_for_dataset_invalid_dataset(
    dataset_config,
    manifest,
    manifest_path,
    manifest_s3uri,
    position,
    expected_path,
    expected_s3uri,
):
    if expected_path != "invalid" and expected_s3uri != "invalid":
        pytest.skip("Skipping valid parameter combination for testing invalid combinations")

    dataset_config.name = "dataset_name"
    dataset_config.zarr_positions = [1, 3, 5]
    dataset_config.duration = 10

    manifest.locations["dataset_name"] = ExtraLocation(path=manifest_path, s3uri=manifest_s3uri)

    with pytest.raises(ValueError):
        get_extra_location_for_dataset(manifest, dataset_config, position)
