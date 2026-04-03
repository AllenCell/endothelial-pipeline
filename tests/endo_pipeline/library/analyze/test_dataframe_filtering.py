import pandas as pd
import pytest

from endo_pipeline.configs import (
    ChannelIndices,
    DatasetConfig,
    FlowCondition,
    PositionAnnotation,
    TimepointAnnotation,
)
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_annotations,
    filter_dataframe_by_flow_condition,
    filter_dataframe_by_track_length,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.settings.column_names import ColumnName as Column


@pytest.fixture()
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
        duration=4,
        time_interval_in_minutes=1.0,
        channel_names=[],
        flow_conditions=[],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        position_annotations={PositionAnnotation.DUST_ARTIFACT: [1]},
        timepoint_annotations={
            TimepointAnnotation.NOT_STEADY_STATE: {1: [[0, 1]], 3: [[0, 1]], 5: [0]},
            TimepointAnnotation.CELL_PILING: {1: [], 3: [[2, 3]], 5: [3]},
            TimepointAnnotation.AUTO_BF_SCOPE_ERROR: {1: [1], 3: [], 5: []},
        },
    )


@pytest.fixture()
def two_condition_dataset():
    """Dataset with two flow conditions: frames 0 to 1 at low shear, frames 2 to
    3 at high shear.
    """
    return DatasetConfig(
        name="unique_dataset_name",
        date="YYYYMMDD",
        original_path="/path/to/original/dataset",
        zarr_positions=[1],
        fmsid="FMS ID",
        barcode="Dataset LabKey barcode",
        cell_lines=["AICS-111"],
        live_or_fixed_sample="live",
        is_timelapse=True,
        microscope="3i",
        objective="20X",
        shear_stress_regime=[],
        pixel_size_xy_in_um=0.0,
        duration=4,
        time_interval_in_minutes=1.0,
        channel_names=[],
        flow_conditions=[
            FlowCondition(start=0, stop=1, shear_stress=1.0),
            FlowCondition(start=2, stop=3, shear_stress=2.0),
        ],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        position_annotations={},
        timepoint_annotations={},
    )


@pytest.fixture
def dataframe():
    timepoints = list(range(4))
    positions = [1, 3, 5]
    positions_tiled = [i for i in positions for _ in timepoints]
    num_rows = len(positions_tiled)
    return pd.DataFrame(
        {
            Column.DATASET: ["unique_dataset_name"] * num_rows,
            Column.POSITION: positions_tiled,
            Column.TIMEPOINT: timepoints * len(positions),
        }
    )


@pytest.mark.parametrize(
    "position_annotations, timepoint_annotations, expected_positions, expected_timepoints",
    [
        (
            PositionAnnotation.DUST_ARTIFACT,
            [TimepointAnnotation.NOT_STEADY_STATE],
            [3] * 2 + [5] * 3,
            [2, 3, 1, 2, 3],
        ),
        (
            PositionAnnotation.DUST_ARTIFACT,
            [TimepointAnnotation.CELL_PILING],
            [3] * 2 + [5] * 3,
            [0, 1, 0, 1, 2],
        ),
        (
            [],
            [TimepointAnnotation.CELL_PILING],
            [1] * 4 + [3] * 2 + [5] * 3,
            [0, 1, 2, 3, 0, 1, 0, 1, 2],
        ),
        (
            [],
            [TimepointAnnotation.NOT_STEADY_STATE],
            [1] * 2 + [3] * 2 + [5] * 3,
            [2, 3, 2, 3, 1, 2, 3],
        ),
        (
            [],
            [TimepointAnnotation.AUTO_BF_SCOPE_ERROR],
            [1] * 3 + [3] * 4 + [5] * 4,
            [0, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ),
        ([], None, [1] * 2 + [5] * 2, [2, 3, 1, 2]),
        ([PositionAnnotation.DUST_ARTIFACT], [], [3] * 4 + [5] * 4, [0, 1, 2, 3] * 2),
        (None, None, [5, 5], [1, 2]),
    ],
)
def test_filter_dataframe_by_annotations_with_annotations(
    dataframe,
    dataset,
    position_annotations,
    timepoint_annotations,
    expected_positions,
    expected_timepoints,
):
    filtered_df = filter_dataframe_by_annotations(
        dataframe, dataset, position_annotations, timepoint_annotations
    )
    assert filtered_df[Column.POSITION].tolist() == expected_positions
    assert filtered_df[Column.TIMEPOINT].tolist() == expected_timepoints


def test_filter_dataframe_by_annotations_without_annotations(dataframe, dataset):
    filtered_df = filter_dataframe_by_annotations(dataframe, dataset, [], [])
    assert filtered_df[Column.POSITION].tolist() == dataframe[Column.POSITION].tolist()
    assert filtered_df[Column.TIMEPOINT].tolist() == dataframe[Column.TIMEPOINT].tolist()


@pytest.mark.parametrize(
    "dataframe, minimum_track_length, expected_filtered_dataframe",
    [
        (
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [1, 2, 3, 4, 5],
                    "other_column": [10, 20, 30, 40, 50],
                }
            ),
            3,
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [3, 4, 5],
                    "other_column": [30, 40, 50],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [0, 1, 2],
                    "other_column": [10, 20, 30],
                }
            ),
            1,
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [1, 2],
                    "other_column": [20, 30],
                }
            ),
        ),
    ],
)
def test_filter_dataframe_by_track_length_valid_column(
    dataframe, minimum_track_length, expected_filtered_dataframe
):
    filtered_df = filter_dataframe_by_track_length(dataframe, minimum_track_length)
    pd.testing.assert_frame_equal(filtered_df, expected_filtered_dataframe, check_like=True)


@pytest.mark.parametrize(
    "dataframe, minimum_track_length",
    [
        (
            pd.DataFrame(
                {
                    "some_other_column": [1, 2, 3],
                }
            ),
            3,
        ),
    ],
)
def test_filter_dataframe_by_track_length_invalid_column(dataframe, minimum_track_length):
    with pytest.raises(ValueError):
        filter_dataframe_by_track_length(dataframe, minimum_track_length)


def test_filter_dataframe_by_track_length_all_filtered_out():
    # if all tracks are shorter than minimum_track_length, should return empty dataframe with same columns
    dataframe = pd.DataFrame(
        {
            Column.TRACK_LENGTH: [1, 2, 3],
            "other_column": [10, 20, 30],
        }
    )
    with pytest.raises(ValueError):
        filter_dataframe_by_track_length(dataframe, 4)


def test_filter_dataframe_to_steady_state_removes_not_steady_state_timepoints(dataframe, dataset):
    # position_annotations=[] → no position filtering, all positions (1, 3, 5) are kept
    # timepoint_annotations=[NOT_STEADY_STATE] → removes timepoints 0,1 from pos 1 and pos 3,
    #   and timepoint 0 from pos 5
    filtered_df = filter_dataframe_to_steady_state(dataframe, dataset)
    assert filtered_df[Column.POSITION].tolist() == [1, 1, 3, 3, 5, 5, 5]
    assert filtered_df[Column.TIMEPOINT].tolist() == [2, 3, 2, 3, 1, 2, 3]


def test_filter_dataframe_to_steady_state_keeps_all_timepoints_when_no_not_steady_state_annotations(
    dataframe,
):
    # Dataset with no NOT_STEADY_STATE annotations and no position annotations → full dataframe returned
    dataset_no_annotations = DatasetConfig(
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
        duration=4,
        time_interval_in_minutes=1.0,
        channel_names=[],
        flow_conditions=[],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        position_annotations={},
        timepoint_annotations={},
    )
    filtered_df = filter_dataframe_to_steady_state(dataframe, dataset_no_annotations)
    assert filtered_df[Column.POSITION].tolist() == dataframe[Column.POSITION].tolist()
    assert filtered_df[Column.TIMEPOINT].tolist() == dataframe[Column.TIMEPOINT].tolist()


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame(
            {
                Column.DATASET: ["unique_dataset_name"],
                Column.TIMEPOINT: [0],
                # Column.POSITION missing
            }
        ),
        pd.DataFrame(
            {
                Column.POSITION: [1],
                Column.TIMEPOINT: [0],
                # Column.DATASET missing
            }
        ),
        pd.DataFrame(
            {
                Column.DATASET: ["unique_dataset_name"],
                Column.POSITION: [1],
                # Column.TIMEPOINT missing
            }
        ),
    ],
)
def test_filter_dataframe_to_steady_state_raises_with_missing_required_columns(dataframe, dataset):
    with pytest.raises(ValueError, match="DataFrame must contain column"):
        filter_dataframe_to_steady_state(dataframe, dataset)


def test_filter_by_flow_condition_single_condition_returns_all_rows(dataframe, dataset):
    """When dataset has only one flow condition, all rows should be returned unchanged."""
    single_condition = FlowCondition(start=0, stop=5, shear_stress=1.0)
    dataset.flow_conditions = [single_condition]
    result = filter_dataframe_by_flow_condition(dataframe, dataset, single_condition)
    assert result[Column.TIMEPOINT].tolist() == dataframe[Column.TIMEPOINT].tolist()


def test_filter_by_flow_condition_first_condition_returns_frames_before_change(
    dataframe, two_condition_dataset
):
    """First flow condition should include only frames before the change frame (< 2)."""
    first_condition = two_condition_dataset.flow_conditions[0]
    result = filter_dataframe_by_flow_condition(dataframe, two_condition_dataset, first_condition)
    assert result[Column.TIMEPOINT].tolist() == [0, 1] * 3


def test_filter_by_flow_condition_second_condition_returns_frames_from_change(
    dataframe, two_condition_dataset
):
    """Second flow condition should include only frames from the change frame onward (>= 2)."""
    second_condition = two_condition_dataset.flow_conditions[1]
    result = filter_dataframe_by_flow_condition(dataframe, two_condition_dataset, second_condition)
    assert result[Column.TIMEPOINT].tolist() == [2, 3] * 3


def test_filter_by_flow_condition_returns_copy(dataframe, two_condition_dataset):
    """Returned dataframe should be a copy; mutating it should not affect the original."""
    first_condition = two_condition_dataset.flow_conditions[0]
    result = filter_dataframe_by_flow_condition(dataframe, two_condition_dataset, first_condition)
    result[Column.TIMEPOINT] = -1
    assert dataframe[Column.TIMEPOINT].tolist() == [0, 1, 2, 3] * 3


def test_filter_by_flow_condition_raises_when_flow_condition_not_in_dataset(
    dataframe, two_condition_dataset
):
    """Providing a FlowCondition not present in the dataset config should raise ValueError."""
    unrelated_condition = FlowCondition(start=20, stop=30, shear_stress=99.0)
    with pytest.raises(ValueError, match="does not match any of the flow conditions"):
        filter_dataframe_by_flow_condition(dataframe, two_condition_dataset, unrelated_condition)


@pytest.mark.parametrize(
    "bad_dataframe",
    [
        pd.DataFrame(
            {
                Column.DATASET: ["unique_dataset_name"],
                # Column.TIMEPOINT missing
            }
        ),
        pd.DataFrame(
            {
                Column.TIMEPOINT: [0],
                # Column.DATASET missing
            }
        ),
    ],
)
def test_filter_by_flow_condition_raises_with_missing_required_columns(
    bad_dataframe, two_condition_dataset
):
    """Missing required columns should raise a ValueError."""
    with pytest.raises(ValueError, match="DataFrame must contain column"):
        filter_dataframe_by_flow_condition(
            bad_dataframe,
            two_condition_dataset,
            two_condition_dataset.flow_conditions[0],
        )


def test_filter_by_flow_condition_raises_when_dataset_name_mismatch(two_condition_dataset):
    """A dataframe whose dataset name differs from the config name should raise an error."""
    mismatched_df = pd.DataFrame(
        {
            Column.DATASET: ["different_dataset_name"] * 5,
            Column.TIMEPOINT: list(range(5)),
        }
    )
    with pytest.raises(ValueError):
        filter_dataframe_by_flow_condition(
            mismatched_df,
            two_condition_dataset,
            two_condition_dataset.flow_conditions[0],
        )
