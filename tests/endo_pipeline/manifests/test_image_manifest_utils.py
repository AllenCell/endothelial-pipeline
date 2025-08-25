from pathlib import Path

import pytest

from endo_pipeline.configs import ChannelIndices, DatasetConfig, FlowCondition
from endo_pipeline.manifests.image_manifest import ImageLocation, ImageManifest
from endo_pipeline.manifests.image_manifest_utils import (
    get_image_location_for_dataset,
    list_datasets_with_images,
)


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
        original_path="",
        zarr_path="",
        zarr_positions=[1, 3, 5],
        fmsid="",
        barcode="",
        cell_lines=[""],
        live_or_fixed_sample="live",
        is_timelapse=True,
        microscope="3i",
        objective="20X",
        shear_stress_regime="",
        pixel_size_xy_in_um=0.0,
        duration=0,
        time_interval_in_minutes=0.0,
        flow_conditions=[FlowCondition(start=0, stop=0, shear_stress=0.0)],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
    )


@pytest.fixture
def mock_load_dataset_config(mocker):
    def _mocker(dataset_config):
        config_mock = mocker.patch(
            "endo_pipeline.manifests.image_manifest_utils.load_dataset_config"
        )
        config_mock.return_value = dataset_config

    return _mocker


def test_list_datasets_with_images_with_valid_locations(manifest):
    manifest.locations = {
        "dataset_one": ImageLocation(path=Path("/path/to/dataset_one/seg.ome.tiff")),
        "dataset_two": ImageLocation(path=Path("/path/to/dataset_two/seg.ome.tiff")),
        "dataset_three": ImageLocation(path=Path("/path/to/dataset_three/seg.ome.tiff")),
    }

    datasets = list_datasets_with_images(manifest)

    assert datasets == ["dataset_one", "dataset_two", "dataset_three"]


def test_list_datasets_with_images_with_invalid_location(manifest):
    manifest.locations = {
        "dataset_one": ImageLocation(path=Path("/path/to/dataset_one/seg.ome.tiff")),
        "dataset_two": ImageLocation(),
        "dataset_three": ImageLocation(path=Path("/path/to/dataset_three/seg.ome.tiff")),
    }

    datasets = list_datasets_with_images(manifest)

    assert datasets == ["dataset_one", "dataset_three"]


@pytest.mark.parametrize(
    "dataset_name,manifest_path,position,timepoint,expected_path",
    [
        (
            "no_position_no_timepoint",
            "/path/to/seg/seg.ome.tiff",
            None,
            None,
            "/path/to/seg/seg.ome.tiff",
        ),
        (
            "valid_position_no_timepoint",
            "/path/to/seg/P{{position}}_seg.ome.tiff",
            1,
            None,
            "/path/to/seg/P1_seg.ome.tiff",
        ),
        (
            "valid_position_unused_timepoint",
            "/path/to/seg/P{{position}}_seg.ome.tiff",
            1,
            1,
            "/path/to/seg/P1_seg.ome.tiff",
        ),
        (
            "no_position_valid_timepoint",
            "/path/to/seg/T{{timepoint}}_seg.ome.tiff",
            None,
            9,
            "/path/to/seg/T9_seg.ome.tiff",
        ),
        (
            "unused_position_valid_timepoint",
            "/path/to/seg/T{{timepoint}}_seg.ome.tiff",
            3,
            9,
            "/path/to/seg/T9_seg.ome.tiff",
        ),
        (
            "valid_position_valid_timepoint",
            "/path/to/seg/P{{position}}_T{{timepoint}}_seg.ome.tiff",
            3,
            5,
            "/path/to/seg/P3_T5_seg.ome.tiff",
        ),
    ],
)
def test_get_image_location_for_dataset_valid_dataset_valid_arguments(
    mock_load_dataset_config,
    dataset_config,
    manifest,
    dataset_name,
    manifest_path,
    position,
    timepoint,
    expected_path,
):
    dataset_config.name = dataset_name
    dataset_config.zarr_positions = [1, 3, 5]
    dataset_config.duration = 10

    mock_load_dataset_config(dataset_config)
    manifest.locations[dataset_name] = ImageLocation(path=manifest_path)

    location = get_image_location_for_dataset(manifest, dataset_name, position, timepoint)

    assert location.path.as_posix() == expected_path


@pytest.mark.parametrize(
    "dataset_name,manifest_path,position,timepoint",
    [
        (
            "invalid_position_no_timepoint",
            "/path/to/seg/P{{position}}_seg.ome.tiff",
            2,
            None,
        ),
        (
            "no_position_invalid_negative_timepoint",
            "/path/to/seg/T{{timepoint}}_seg.ome.tiff",
            None,
            -1,
        ),
        (
            "no_position_invalid_positive_timepoint",
            "/path/to/seg/T{{timepoint}}_seg.ome.tiff",
            None,
            11,
        ),
        (
            "missing_position_no_timepoint",
            "/path/to/seg/P{{position}}_seg.ome.tiff",
            None,
            None,
        ),
        (
            "no_position_missing_timepoint",
            "/path/to/seg/T{{timepoint}}_seg.ome.tiff",
            None,
            None,
        ),
    ],
)
def test_get_image_location_for_dataset_valid_dataset_invalid_arguments(
    mock_load_dataset_config,
    dataset_config,
    manifest,
    dataset_name,
    manifest_path,
    position,
    timepoint,
):
    dataset_config.name = dataset_name
    dataset_config.zarr_positions = [1, 3, 5]
    dataset_config.duration = 10

    mock_load_dataset_config(dataset_config)
    manifest.locations[dataset_name] = ImageLocation(path=manifest_path)

    with pytest.raises(ValueError):
        get_image_location_for_dataset(manifest, dataset_name, position, timepoint)


def test_get_image_location_for_dataset_valid_dataset_no_path(
    mock_load_dataset_config, dataset_config, manifest
):
    dataset_config.name = "no_seg_path"
    dataset_config.zarr_positions = [1, 3, 5]
    dataset_config.duration = 10

    mock_load_dataset_config(dataset_config)
    manifest.locations[dataset_config.name] = ImageLocation()

    location = get_image_location_for_dataset(manifest, dataset_config.name, 10, 10)

    assert location.path is None


def test_get_image_location_for_dataset_invalid_dataset(manifest):
    with pytest.raises(KeyError):
        get_image_location_for_dataset(manifest, "invalid_dataset")
