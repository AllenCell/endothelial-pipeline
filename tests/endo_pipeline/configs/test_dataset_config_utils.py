from pathlib import Path

import bioio
import pytest

from endo_pipeline.configs import (
    ChannelIndices,
    DatasetConfig,
    FlowCondition,
    PositionAnnotation,
    TimepointAnnotation,
)
from endo_pipeline.configs.dataset_config_utils import (
    get_annotated_positions,
    get_annotated_timepoints_for_position,
    get_duration_at_flow,
    get_flow_at_frame,
    get_frame_after_flow_change,
    get_frame_before_flow_change,
    get_position_integer_from_zarr_file_path,
    get_position_string_from_zarr_file_path,
    get_start_of_steady_state_for_position,
    get_unannotated_positions,
    get_unannotated_timepoints_for_position,
)


@pytest.fixture
def dataset():
    return DatasetConfig(
        name="unique_dataset_name",
        date="YYYYMMDD",
        original_path="/path/to/original/dataset",
        zarr_positions=[1, 3, 5],
        fmsid="FMS ID",
        barcode="Dataset LabKey barcode",
        cell_lines=["AICS-111", "AICS-222"],
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


@pytest.fixture
def mock_get_zarr_location_for_position(mocker):
    def _mocker(dataset_config, image_locations):
        manifest_mock = mocker.patch("endo_pipeline.manifests.get_zarr_location_for_position")
        manifest_mock.side_effect = lambda x, p: image_locations[p] if x == dataset_config else None

    return _mocker


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
    "annotations,positions",
    [(None, [1, 2, 3]), ([PositionAnnotation.DUST_ARTIFACT], [1, 2, 3]), ([], [])],
)
def test_get_annotated_positions_with_annotations(dataset, annotations, positions):
    dataset.position_annotations = {PositionAnnotation.DUST_ARTIFACT: [1, 2, 3]}

    assert get_annotated_positions(dataset, annotations) == positions


def test_get_annotated_positions_no_annotations(dataset):
    assert get_annotated_positions(dataset, None) == []


@pytest.mark.parametrize(
    "annotations,positions",
    [(None, [5]), ([PositionAnnotation.DUST_ARTIFACT], [5]), ([], [1, 3, 5])],
)
def test_get_unannotated_positions_with_annotations(dataset, annotations, positions):
    dataset.position_annotations = {PositionAnnotation.DUST_ARTIFACT: [1, 2, 3]}

    assert get_unannotated_positions(dataset, annotations) == positions


def test_get_unannotated_positions_no_annotations(dataset):
    assert get_unannotated_positions(dataset, None) == dataset.zarr_positions


@pytest.mark.parametrize(
    "position,annotations,timepoints",
    [
        (0, None, [1, 2, 3, 7, 8, 9, 13, 14, 15]),
        (1, None, [4, 5, 6, 10, 11, 12, 13]),
        (2, None, [16, 17, 18]),
        (0, [TimepointAnnotation.BF_SCOPE_ERROR], [1, 2, 3]),
        (1, [TimepointAnnotation.BF_SCOPE_ERROR], [4, 5, 6]),
        (2, [TimepointAnnotation.BF_SCOPE_ERROR], []),
        (0, [TimepointAnnotation.GFP_SCOPE_ERROR], [7, 8, 9]),
        (1, [TimepointAnnotation.GFP_SCOPE_ERROR], [10, 11, 12, 13]),
        (2, [TimepointAnnotation.GFP_SCOPE_ERROR], []),
        (0, [TimepointAnnotation.XY_SHIFT], [13, 14, 15]),
        (1, [TimepointAnnotation.XY_SHIFT], []),
        (2, [TimepointAnnotation.XY_SHIFT], [16, 17, 18]),
        (0, [], []),
        (1, [], []),
        (2, [], []),
    ],
)
def test_get_annotated_timepoints_for_position_with_annotations(
    dataset, position, annotations, timepoints
):
    dataset.timepoint_annotations = {
        TimepointAnnotation.BF_SCOPE_ERROR: {
            0: [1, 2, 3],
            1: [[4, 6]],
        },
        TimepointAnnotation.GFP_SCOPE_ERROR: {
            0: [7, 8, 9],
            1: [[11, 13], 10],
        },
        TimepointAnnotation.XY_SHIFT: {
            0: [13, 14, 15],
            1: [],
            2: [16, 17, 18],
        },
    }

    assert get_annotated_timepoints_for_position(dataset, position, annotations) == timepoints


def test_get_annotated_timepoints_for_position_no_annotations(dataset):
    assert get_annotated_timepoints_for_position(dataset, 0, None) == []


# test get_start_of_steady_state_for_position
@pytest.mark.parametrize(
    "position,expected_timepoint",
    [(0, 5), (1, 6), (2, None)],
)
def test_get_start_of_steady_state_for_position(dataset, position, expected_timepoint):
    dataset.duration = 10
    dataset.timepoint_annotations = {
        TimepointAnnotation.NOT_STEADY_STATE: {
            0: [[0, 4]],
            1: [[0, 5]],
            2: [],
        },
    }

    assert get_start_of_steady_state_for_position(dataset, position) == expected_timepoint


def test_get_start_of_steady_state_for_position_exceeds_duration(dataset):
    position = 0
    expected_timepoint = None
    dataset.duration = 4
    dataset.timepoint_annotations = {
        TimepointAnnotation.NOT_STEADY_STATE: {
            0: [[0, 4]],
        },
    }

    assert get_start_of_steady_state_for_position(dataset, position) == expected_timepoint


@pytest.mark.parametrize(
    "position,annotations,timepoints",
    [
        (0, None, [0, 4]),
        (1, None, [7, 8, 9]),
        (2, None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, [TimepointAnnotation.BF_SCOPE_ERROR], [0, 4, 5, 6, 7, 8, 9]),
        (1, [TimepointAnnotation.BF_SCOPE_ERROR], [0, 1, 2, 3, 7, 8, 9]),
        (2, [TimepointAnnotation.BF_SCOPE_ERROR], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, [TimepointAnnotation.GFP_SCOPE_ERROR], [0, 1, 2, 3, 4, 5, 6]),
        (1, [TimepointAnnotation.GFP_SCOPE_ERROR], [4, 5, 6, 7, 8, 9]),
        (2, [TimepointAnnotation.GFP_SCOPE_ERROR], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, [], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (1, [], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (2, [], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, [TimepointAnnotation.BF_SCOPE_ERROR, TimepointAnnotation.NOT_STEADY_STATE], [0, 4]),
    ],
)
def test_get_unannotated_timepoints_for_position_with_annotations(
    dataset, position, annotations, timepoints
):
    dataset.duration = 10
    dataset.timepoint_annotations = {
        TimepointAnnotation.BF_SCOPE_ERROR: {
            0: [1, 2, 3],
            1: [[4, 6]],
        },
        TimepointAnnotation.GFP_SCOPE_ERROR: {
            0: [7, 8, 9],
            1: [[1, 3], 0],
            2: [],
        },
        TimepointAnnotation.NOT_STEADY_STATE: {
            0: [[5, 9]],
        },
    }

    assert get_unannotated_timepoints_for_position(dataset, position, annotations) == timepoints


def test_get_unannotated_timepoints_for_position_no_annotations(dataset):
    assert get_unannotated_timepoints_for_position(dataset, 0, None) == []


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
