from pathlib import Path

import pytest
from bioio import BioImage

from src.endo_pipeline.configs import (
    DatasetConfig,
    load_dataset_collection_config,
    load_dataset_config,
)
from src.endo_pipeline.configs.dataset_config_utils import (
    get_available_channels_for_all_positions,
    get_available_channels_for_position,
    get_available_zarr_files,
    get_channel_indices_for_all_positions,
    get_channel_indices_for_position,
    get_nuclear_prediction_path,
    get_specific_channel_order,
    get_zarr_file_for_position,
    make_sample_type_objective_microscope_collection,
)


@pytest.fixture
def dataset():
    return DatasetConfig(
        name="unique_dataset_name",
        original_path="/path/to/original/dataset",
        zarr_path="/path/to/zarr/dataset",
        zarr_positions=[1, 3, 5],
        fmsid="FMS ID",
        barcode="Dataset LabKey barcode",
        cell_lines=["AICS-111", "AICS-222"],
        live_or_fixed_sample="live",
        is_timelapse=True,
        microscope="3i",
        shear_stress_regime="Shear stress regime the dataset was collected under",
        pixel_size_xy_in_um=0.0,
        duration=0,
        time_interval_in_minutes=0.0,
        flow=[(0, 0, 0.0)],
        n_total_positions=0,
        brightfield_channel_index=0,
        channel_488_index=0,
    )


@pytest.fixture(autouse=True)
def zarr_files(mocker):
    zarr_p1_mock = mocker.MagicMock(spec=BioImage)
    zarr_p1_mock.channel_names = ["Channel1", "Channel2"]

    zarr_p3_mock = mocker.MagicMock(spec=BioImage)
    zarr_p3_mock.channel_names = ["Channel1", "Channel2", "Channel3"]

    zarr_p5_mock = mocker.MagicMock(spec=BioImage)
    zarr_p5_mock.channel_names = ["Channel1", "Channel3", "Channel4"]

    files = {
        Path("/path/to/zarr/dataset/dataset_P1.ome.zarr"): zarr_p1_mock,
        Path("/path/to/zarr/dataset/dataset_P3.ome.zarr"): zarr_p3_mock,
        Path("/path/to/zarr/dataset/dataset_P5.ome.zarr"): zarr_p5_mock,
    }

    mock = mocker.patch("src.endo_pipeline.configs.dataset_config_utils.BioImage")
    mock.side_effect = lambda arg: files[arg]


def test_get_available_zarr_files(dataset):
    zarr_files = get_available_zarr_files(dataset)

    expected = [
        Path("/path/to/zarr/dataset/dataset_P1.ome.zarr"),
        Path("/path/to/zarr/dataset/dataset_P3.ome.zarr"),
        Path("/path/to/zarr/dataset/dataset_P5.ome.zarr"),
    ]

    assert zarr_files == expected


def test_get_zarr_file_for_position_valid(dataset):
    zarr_file = get_zarr_file_for_position(dataset, position=3)

    assert zarr_file == Path("/path/to/zarr/dataset/dataset_P3.ome.zarr")


def test_get_zarr_file_for_position_invalid(dataset):
    with pytest.raises(ValueError):
        get_zarr_file_for_position(dataset, position=4)


def test_get_available_channels_for_all_positions(dataset):
    channels = get_available_channels_for_all_positions(dataset)

    assert channels == {
        1: ["Channel1", "Channel2"],
        3: ["Channel1", "Channel2", "Channel3"],
        5: ["Channel1", "Channel3", "Channel4"],
    }


@pytest.mark.parametrize(
    "position,expected",
    [
        (1, ["Channel1", "Channel2"]),
        (3, ["Channel1", "Channel2", "Channel3"]),
        (5, ["Channel1", "Channel3", "Channel4"]),
    ],
)
def test_get_available_channels_for_position(dataset, position, expected):
    channels = get_available_channels_for_position(dataset, position)

    assert channels == expected


def test_get_channel_indices_for_all_positions(dataset):
    channel_names = ["Channel3", "Channel1", "Channel2"]

    channels = get_channel_indices_for_all_positions(dataset, channel_names)

    assert channels == {
        1: [None, 0, 1],
        3: [2, 0, 1],
        5: [1, 0, None],
    }


@pytest.mark.parametrize(
    "position,expected",
    [
        (1, [None, 1]),
        (3, [2, 1]),
        (5, [1, None]),
    ],
)
def test_get_channel_indices_for_position(dataset, position, expected):
    channel_names = ["Channel3", "Channel2"]

    indices = get_channel_indices_for_position(dataset, position, channel_names)

    assert indices == expected


def test_get_specific_channel_order_no_null_channels(dataset):
    dataset.brightfield_channel_index = 1
    dataset.channel_488_index = 2
    dataset.channel_405_index = 3
    dataset.channel_561_index = 4
    dataset.channel_640_index = 5

    channel_order = get_specific_channel_order(dataset)

    assert channel_order == (2, 1, 3, 4, 5)


def test_get_specific_channel_order_with_null_channels(dataset):
    dataset.brightfield_channel_index = 1
    dataset.channel_488_index = 2
    dataset.channel_405_index = None
    dataset.channel_561_index = None
    dataset.channel_640_index = 5

    channel_order = get_specific_channel_order(dataset)

    assert channel_order == (2, 1, None, None, 5)


@pytest.mark.parametrize(
    "nuc_seg_type,position,expected",
    [
        ("label_free", 1, "/path/to/label/free/seg/P1"),
        ("stain", 2, "/path/to/stain/seg/P2"),
    ],
)
def test_get_nuclear_prediction_path_valid_paths(dataset, nuc_seg_type, position, expected):
    dataset.nuclear_label_free_seg_path = "/path/to/label/free/seg"
    dataset.nuclear_stain_seg_path = "/path/to/stain/seg/"

    nuclear_prediction_path = get_nuclear_prediction_path(dataset, position, nuc_seg_type)
    assert nuclear_prediction_path.as_posix() == expected


@pytest.mark.parametrize("nuc_seg_type", ["label_free", "stain", "invalid"])
def test_get_nuclear_prediction_path_invalid_paths(dataset, nuc_seg_type):
    dataset.nuclear_label_free_seg_path = None
    dataset.nuclear_stain_seg_path = None

    with pytest.raises(ValueError):
        get_nuclear_prediction_path(dataset, 0, nuc_seg_type)


def test_make_live_20X_objective_3i_microscope_dataset_collection():
    """
    Test the creation of a dataset collection
    for live samples with 20X objective and 3i microscope.
    """
    dataset_collection = make_sample_type_objective_microscope_collection(
        sample_type="live",
        objective="20X",
        microscope="3i",
    )
    dataset_configs = [
        load_dataset_config(dataset_name) for dataset_name in dataset_collection.datasets
    ]
    assert all(
        dataset_config.live_or_fixed_sample == "live"
        and "20X" in dataset_config.name
        and dataset_config.microscope == "3i"
        for dataset_config in dataset_configs
    )


def test_loaded_live_20X_objective_3i_microscope_collection():
    """
    Compare the current live 20X objective 3i microscope collection
    with the one generated by the function.
    """
    dataset_collection = make_sample_type_objective_microscope_collection(
        sample_type="live",
        objective="20X",
        microscope="3i",
    )
    loaded_dataset_collection = load_dataset_collection_config(dataset_collection.name)
    assert loaded_dataset_collection.datasets == dataset_collection.datasets
