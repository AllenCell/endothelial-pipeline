from pathlib import Path

import pytest

from endo_pipeline.configs import ChannelIndices, DatasetConfig
from endo_pipeline.manifests.image_manifest import ImageLocation, ImageManifest
from endo_pipeline.manifests.image_manifest_utils import (
    get_available_zarr_locations,
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    list_datasets_with_images,
)
from endo_pipeline.settings.manifest_names import ZARR_IMAGE_MANIFEST_NAME


@pytest.fixture
def manifest():
    return ImageManifest(
        name="unique_image_manifest_name",
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
def mock_load_image_manifest(mocker):
    def _mocker(manifest_name, image_manifest):
        manifest_mock = mocker.patch(
            "endo_pipeline.manifests.image_manifest_utils.load_image_manifest"
        )
        manifest_mock.side_effect = lambda x: image_manifest if x == manifest_name else None

    return _mocker


def test_list_datasets_with_images_with_valid_locations(manifest):
    manifest.locations = {
        "dataset_one": ImageLocation(path=Path("dataset_one.ome.tiff")),
        "dataset_two": ImageLocation(path=Path("dataset_two.ome.tiff"), s3uri="s3://dataset_two"),
        "dataset_three": ImageLocation(s3uri="s3://dataset_three"),
    }

    datasets = list_datasets_with_images(manifest)

    assert datasets == ["dataset_one", "dataset_two", "dataset_three"]


def test_list_datasets_with_images_with_invalid_location(manifest):
    manifest.locations = {
        "dataset_one": ImageLocation(path=Path("dataset_one.ome.tiff")),
        "dataset_two": ImageLocation(),
        "dataset_three": ImageLocation(s3uri="s3://dataset_three"),
    }

    datasets = list_datasets_with_images(manifest)

    assert datasets == ["dataset_one", "dataset_three"]


@pytest.fixture
def expected_path(timepoint, position, manifest_path):
    if manifest_path is None:
        return None
    elif position is None and "{{position}}" in str(manifest_path):
        return "invalid"
    elif timepoint is None and "{{timepoint}}" in str(manifest_path):
        return "invalid"
    elif (timepoint == 11 or timepoint == -1) and "{{timepoint}}" in str(manifest_path):
        return "invalid"
    elif (position == 4 or position == -1) and "{{position}}" in str(manifest_path):
        return "invalid"
    else:
        expected_path = manifest_path.replace("{{timepoint}}", str(timepoint))
        expected_path = expected_path.replace("{{position}}", str(position))
        return Path(expected_path)


@pytest.fixture
def expected_s3uri(timepoint, position, manifest_s3uri):
    if manifest_s3uri is None:
        return None
    elif position is None and "{{position}}" in manifest_s3uri:
        return "invalid"
    elif timepoint is None and "{{timepoint}}" in manifest_s3uri:
        return "invalid"
    elif (timepoint == 11 or timepoint == -1) and "{{timepoint}}" in manifest_s3uri:
        return "invalid"
    elif (position == 4 or position == -1) and "{{position}}" in manifest_s3uri:
        return "invalid"
    else:
        expected_s3uri = manifest_s3uri.replace("{{timepoint}}", str(timepoint))
        expected_s3uri = expected_s3uri.replace("{{position}}", str(position))
        return expected_s3uri


@pytest.mark.parametrize(
    "timepoint",
    [None, 5, 11, -1],
    ids=["no_timepoint", "valid_timepoint", "invalid_timepoint", "negative_timepoint"],
)
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
        "/path/to/seg/T{{timepoint}}_seg.ome.tiff",
        "/path/to/seg/P{{position}}_seg.ome.tiff",
        "/path/to/seg/T{{timepoint}}_P{{position}}_seg.ome.tiff",
    ],
    ids=lambda path: (
        "no_path"
        if path is None
        else "_".join(
            [
                "path",
                "with" if "{{timepoint}}" in path else "without",
                "timepoint_placeholder",
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
        "s3://bucket-name/T{{timepoint}}_seg.ome.tiff",
        "s3://bucket-name/P{{position}}_seg.ome.tiff",
        "s3://bucket-name/T{{timepoint}}_P{{position}}_seg.ome.tiff",
    ],
    ids=lambda s3uri: (
        "no_s3uri"
        if s3uri is None
        else "_".join(
            [
                "s3uri",
                "with" if "{{timepoint}}" in s3uri else "without",
                "timepoint_placeholder",
                "with" if "{{position}}" in s3uri else "without",
                "position_placeholder",
            ]
        )
    ),
)
def test_get_image_location_for_dataset_valid_dataset(
    dataset_config,
    manifest,
    manifest_path,
    manifest_s3uri,
    timepoint,
    position,
    expected_path,
    expected_s3uri,
):
    if expected_path == "invalid" or expected_s3uri == "invalid":
        pytest.skip("Skipping invalid parameter combination for testing valid combinations")

    dataset_config.name = "dataset_name"
    dataset_config.zarr_positions = [1, 3, 5]
    dataset_config.duration = 10

    manifest.locations["dataset_name"] = ImageLocation(path=manifest_path, s3uri=manifest_s3uri)

    location = get_image_location_for_dataset(
        manifest, dataset_config, timepoint=timepoint, position=position
    )

    if manifest_path is None:
        assert location.path is None
    else:
        assert location.path == expected_path

    if manifest_s3uri is None:
        assert location.s3uri is None
    else:
        assert location.s3uri == expected_s3uri


@pytest.mark.parametrize(
    "timepoint",
    [None, 5, 11, -1],
    ids=["no_timepoint", "valid_timepoint", "invalid_timepoint", "negative_timepoint"],
)
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
        "/path/to/seg/T{{timepoint}}_seg.ome.tiff",
        "/path/to/seg/P{{position}}_seg.ome.tiff",
        "/path/to/seg/T{{timepoint}}_P{{position}}_seg.ome.tiff",
    ],
    ids=lambda path: (
        "no_path"
        if path is None
        else "_".join(
            [
                "path",
                "with" if "{{timepoint}}" in path else "without",
                "timepoint_placeholder",
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
        "s3://bucket-name/T{{timepoint}}_seg.ome.tiff",
        "s3://bucket-name/P{{position}}_seg.ome.tiff",
        "s3://bucket-name/T{{timepoint}}_P{{position}}_seg.ome.tiff",
    ],
    ids=lambda s3uri: (
        "no_s3uri"
        if s3uri is None
        else "_".join(
            [
                "s3uri",
                "with" if "{{timepoint}}" in s3uri else "without",
                "timepoint_placeholder",
                "with" if "{{position}}" in s3uri else "without",
                "position_placeholder",
            ]
        )
    ),
)
def test_get_image_location_for_dataset_invalid_dataset(
    dataset_config,
    manifest,
    manifest_path,
    manifest_s3uri,
    timepoint,
    position,
    expected_path,
    expected_s3uri,
):
    if expected_path != "invalid" and expected_s3uri != "invalid":
        pytest.skip("Skipping valid parameter combination for testing invalid combinations")

    dataset_config.name = "dataset_name"
    dataset_config.zarr_positions = [1, 3, 5]
    dataset_config.duration = 10

    manifest.locations["dataset_name"] = ImageLocation(path=manifest_path, s3uri=manifest_s3uri)

    with pytest.raises(ValueError):
        get_image_location_for_dataset(manifest, dataset_config, position, timepoint)


def test_get_zarr_location_for_position(mock_load_image_manifest, dataset_config):
    dataset_name = "dataset_name"
    dataset_config.name = dataset_name

    image_manifest = ImageManifest(
        name=ZARR_IMAGE_MANIFEST_NAME,
        workflow="",
        locations={dataset_name: ImageLocation(path=Path("path/to/zarr_{{position}}.ome.zarr"))},
    )

    mock_load_image_manifest(ZARR_IMAGE_MANIFEST_NAME, image_manifest)

    location = get_zarr_location_for_position(dataset_config, 3)

    assert location.path.as_posix() == "path/to/zarr_3.ome.zarr"


def test_get_available_zarr_locations(mock_load_image_manifest, dataset_config):
    dataset_name = "dataset_name"
    dataset_config.name = dataset_name

    image_manifest = ImageManifest(
        name=ZARR_IMAGE_MANIFEST_NAME,
        workflow="",
        locations={dataset_name: ImageLocation(path=Path("path/to/zarr_{{position}}.ome.zarr"))},
    )

    mock_load_image_manifest(ZARR_IMAGE_MANIFEST_NAME, image_manifest)

    expected_locations = [
        Path("path/to/zarr_1.ome.zarr"),
        Path("path/to/zarr_3.ome.zarr"),
        Path("path/to/zarr_5.ome.zarr"),
    ]

    zarr_locations = get_available_zarr_locations(dataset_config)

    for location, expected_path in zip(zarr_locations, expected_locations, strict=True):
        assert location.path.as_posix() == expected_path.as_posix()
