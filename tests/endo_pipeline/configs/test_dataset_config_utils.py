from pathlib import Path

import bioio
import pytest

from endo_pipeline.configs import ChannelIndices, DatasetConfig, FlowCondition, load_dataset_config
from endo_pipeline.configs.dataset_config_utils import (
    get_available_channels_for_all_positions,
    get_available_channels_for_position,
    get_available_zarr_files,
    get_channel_indices_for_all_positions,
    get_channel_indices_for_position,
    get_duration_at_flow,
    get_flow_at_frame,
    get_frame_after_flow_change,
    get_frame_before_flow_change,
    get_position_integer_from_zarr_file_path,
    get_position_string_from_zarr_file_path,
    get_zarr_file_for_position,
    make_filtered_dataset_collection,
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
        objective="20X",
        shear_stress_regime="Shear stress regime the dataset was collected under",
        pixel_size_xy_in_um=0.0,
        duration=0,
        time_interval_in_minutes=0.0,
        flow=[(0, 0, 0.0)],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
    )


@pytest.fixture(autouse=True)
def zarr_files(mocker):
    zarr_p1_mock = mocker.MagicMock(spec=bioio.BioImage)
    zarr_p1_mock.channel_names = ["Channel1", "Channel2"]

    zarr_p3_mock = mocker.MagicMock(spec=bioio.BioImage)
    zarr_p3_mock.channel_names = ["Channel1", "Channel2", "Channel3"]

    zarr_p5_mock = mocker.MagicMock(spec=bioio.BioImage)
    zarr_p5_mock.channel_names = ["Channel1", "Channel3", "Channel4"]

    files = {
        Path("/path/to/zarr/dataset/dataset_P1.ome.zarr"): zarr_p1_mock,
        Path("/path/to/zarr/dataset/dataset_P3.ome.zarr"): zarr_p3_mock,
        Path("/path/to/zarr/dataset/dataset_P5.ome.zarr"): zarr_p5_mock,
    }

    bioimage_mock = mocker.MagicMock()
    bioimage_mock.side_effect = lambda arg: files[arg]

    mocker.patch.object(bioio, "BioImage", bioimage_mock)


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


def test_get_frame_before_flow_change_valid_flow_condition(dataset):
    dataset.flow_conditions = [
        FlowCondition(start=0, stop=5, shear_stress=1),
        FlowCondition(start=6, stop=10, shear_stress=2),
    ]

    assert get_frame_before_flow_change(dataset) == 5


@pytest.mark.parametrize("num_conditions", [0, 1, 3, 4])
def test_get_frame_before_flow_change_invalid_flow_condition(dataset, num_conditions):
    dataset.flow_conditions = [FlowCondition(start=0, stop=0, shear_stress=0)] * num_conditions

    assert get_frame_before_flow_change(dataset) is None


def test_get_frame_after_flow_change_valid_flow_condition(dataset):
    dataset.flow_conditions = [
        FlowCondition(start=0, stop=5, shear_stress=1),
        FlowCondition(start=6, stop=10, shear_stress=2),
    ]

    assert get_frame_after_flow_change(dataset) == 6


@pytest.mark.parametrize("num_conditions", [0, 1, 3, 4])
def test_get_frame_after_flow_change_invalid_flow_condition(dataset, num_conditions):
    dataset.flow_conditions = [FlowCondition(start=0, stop=0, shear_stress=0)] * num_conditions

    assert get_frame_after_flow_change(dataset) is None


@pytest.mark.parametrize("frame,expected_flow", [(-5, 1), (-2, 1), (0, 1), (5, 2), (7, 2), (10, 2)])
def test_get_flow_at_frame_valid_frames(dataset, frame, expected_flow):
    dataset.flow_conditions = [
        FlowCondition(start=-5, stop=0, shear_stress=1),
        FlowCondition(start=5, stop=10, shear_stress=2),
    ]

    assert get_flow_at_frame(dataset, frame) == expected_flow


@pytest.mark.parametrize("frame", [-15, -11, 1, 9, 30])
def test_get_flow_at_frame_invalid_frames(dataset, frame):
    dataset.flow_conditions = [
        FlowCondition(start=-10, stop=0, shear_stress=1),
        FlowCondition(start=10, stop=20, shear_stress=2),
        FlowCondition(start=21, stop=29, shear_stress=3),
    ]
    assert get_flow_at_frame(dataset, frame) is None


@pytest.mark.parametrize("flow,expected_duration", [(0, 8), (1, 10), (2, 20), (3, 0)])
def test_get_duration_at_flow(dataset, flow, expected_duration):
    dataset.flow_conditions = [
        FlowCondition(start=-10, stop=0, shear_stress=1),
        FlowCondition(start=10, stop=20, shear_stress=2),
        FlowCondition(start=21, stop=29, shear_stress=0),
        FlowCondition(start=30, stop=40, shear_stress=2),
    ]

    assert get_duration_at_flow(dataset, flow) == expected_duration


@pytest.mark.parametrize(
    "sample_type,objective,microscope",
    [
        ("live", None, None),
        ("fixed", None, None),
        (None, "20X", None),
        (None, "40X", None),
        (None, None, "3i"),
        (None, None, "Nikon"),
        ("live", "20X", "3i"),
        ("live", "20X", "Nikon"),
        ("live", "40X", "3i"),
        ("live", "40X", "Nikon"),
        ("fixed", "20X", "3i"),
        ("fixed", "20X", "Nikon"),
        ("fixed", "40X", "3i"),
        ("fixed", "40X", "Nikon"),
    ],
)
def test_make_filtered_dataset_collection(sample_type, objective, microscope):
    dataset_collection = make_filtered_dataset_collection(
        sample_type=sample_type,
        objective=objective,
        microscope=microscope,
    )

    dataset_configs = [
        load_dataset_config(dataset_name) for dataset_name in dataset_collection.datasets
    ]

    if sample_type is not None:
        assert all(config.live_or_fixed_sample == sample_type for config in dataset_configs)

    if objective is not None:
        assert all(config.objective == objective for config in dataset_configs)

    if microscope is not None:
        assert all(config.microscope == microscope for config in dataset_configs)


@pytest.mark.parametrize(
    "path,expected_position",
    [
        ("/path/to/file/P1.ome.zarr", "P1"),
        ("/path/to/file/before_P2.ome.zarr", "P2"),
        ("/path/to/file/P3_after.ome.zarr", "P3"),
        ("/path/to/file/before_P4_after.ome.zarr", "P4"),
    ],
)
def test_get_position_string_from_zarr_file_path_valid_position(path, expected_position):
    position = get_position_string_from_zarr_file_path(path)

    assert position == expected_position


@pytest.mark.parametrize(
    "path",
    [
        ("/path/to/file/no_position.ome.zarr"),
        ("/path/to/file/P1/position_only_in_path.ome.zarr"),
        ("/path/to/file/lowercase_position_p1.ome.zarr"),
    ],
)
def test_get_position_string_from_zarr_file_path_invalid_position(path):
    with pytest.raises(ValueError):
        get_position_string_from_zarr_file_path(path)


@pytest.mark.parametrize(
    "path,expected_position",
    [
        ("/path/to/file/P1.ome.zarr", 1),
        ("/path/to/file/before_P2.ome.zarr", 2),
        ("/path/to/file/P3_after.ome.zarr", 3),
        ("/path/to/file/before_P14_after.ome.zarr", 14),
    ],
)
def test_get_position_integer_from_zarr_file_path_valid_position(path, expected_position):
    position = get_position_integer_from_zarr_file_path(path)

    assert position == expected_position


@pytest.mark.parametrize(
    "path",
    [
        ("/path/to/file/no_position.ome.zarr"),
        ("/path/to/file/P1/position_only_in_path.ome.zarr"),
        ("/path/to/file/lowercase_position_p1.ome.zarr"),
    ],
)
def test_get_position_integer_from_zarr_file_path_invalid_position(path):
    with pytest.raises(ValueError):
        get_position_integer_from_zarr_file_path(path)
