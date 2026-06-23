import numpy as np
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
    filter_dataframe_by_shear_stress,
    filter_dataframe_by_stability,
    filter_dataframe_by_track_length,
    filter_dataframe_to_binned_value,
    filter_dataframe_to_flow_condition_by_timepoint,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel


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
    dataframe, dataset
):
    # Dataset with no NOT_STEADY_STATE annotations and no position annotations →
    # full dataframe returned
    dataset.position_annotations = {}
    dataset.timepoint_annotations = {}

    filtered_df = filter_dataframe_to_steady_state(dataframe, dataset)
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


def test_filter_to_flow_condition_single_condition_returns_all_rows(dataframe, dataset):
    """When dataset has only one flow condition, all rows should be returned unchanged."""
    single_condition = FlowCondition(start=0, stop=5, shear_stress=1.0)
    dataset.flow_conditions = [single_condition]
    result = filter_dataframe_to_flow_condition_by_timepoint(dataframe, dataset, single_condition)
    assert result[Column.TIMEPOINT].tolist() == dataframe[Column.TIMEPOINT].tolist()


def test_filter_to_flow_condition_first_condition_returns_frames_before_change(
    dataframe, two_condition_dataset
):
    """First flow condition should include only frames before the change frame (< 2)."""
    first_condition = two_condition_dataset.flow_conditions[0]
    result = filter_dataframe_to_flow_condition_by_timepoint(
        dataframe, two_condition_dataset, first_condition
    )
    assert result[Column.TIMEPOINT].tolist() == [0, 1] * 3


def test_filter_to_flow_condition_second_condition_returns_frames_from_change(
    dataframe, two_condition_dataset
):
    """Second flow condition should include only frames from the change frame onward (>= 2)."""
    second_condition = two_condition_dataset.flow_conditions[1]
    result = filter_dataframe_to_flow_condition_by_timepoint(
        dataframe, two_condition_dataset, second_condition
    )
    assert result[Column.TIMEPOINT].tolist() == [2, 3] * 3


def test_filter_to_flow_condition_returns_copy(dataframe, two_condition_dataset):
    """Returned dataframe should be a copy; mutating it should not affect the original."""
    first_condition = two_condition_dataset.flow_conditions[0]
    result = filter_dataframe_to_flow_condition_by_timepoint(
        dataframe, two_condition_dataset, first_condition
    )
    result[Column.TIMEPOINT] = -1
    assert dataframe[Column.TIMEPOINT].tolist() == [0, 1, 2, 3] * 3


def test_filter_to_flow_condition_raises_when_flow_condition_not_in_dataset(
    dataframe, two_condition_dataset
):
    """Providing a FlowCondition not present in the dataset config should raise ValueError."""
    unrelated_condition = FlowCondition(start=20, stop=30, shear_stress=99.0)
    with pytest.raises(ValueError, match="does not match any of the flow conditions"):
        filter_dataframe_to_flow_condition_by_timepoint(
            dataframe, two_condition_dataset, unrelated_condition
        )


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
def test_filter_to_flow_condition_raises_with_missing_required_columns(
    bad_dataframe, two_condition_dataset
):
    """Missing required columns should raise a ValueError."""
    with pytest.raises(ValueError, match="DataFrame must contain column"):
        filter_dataframe_to_flow_condition_by_timepoint(
            bad_dataframe,
            two_condition_dataset,
            two_condition_dataset.flow_conditions[0],
        )


def test_filter_to_flow_condition_raises_when_dataset_name_mismatch(two_condition_dataset):
    """A dataframe whose dataset name differs from the config name should raise an error."""
    mismatched_df = pd.DataFrame(
        {
            Column.DATASET: ["different_dataset_name"] * 5,
            Column.TIMEPOINT: list(range(5)),
        }
    )
    with pytest.raises(ValueError):
        filter_dataframe_to_flow_condition_by_timepoint(
            mismatched_df,
            two_condition_dataset,
            two_condition_dataset.flow_conditions[0],
        )


@pytest.fixture
def binned_dataframe():
    """Simple 1-column dataframe with values spread across three equal bins [0, 1]."""
    return pd.DataFrame({"feat": [0.1, 0.3, 0.6, 0.8, 0.9], "label": list("abcde")})


@pytest.mark.parametrize(
    "value, expected_labels",
    [
        # value 0.2 → bin [0, 0.33): rows 0 and 1
        (0.2, ["a", "b"]),
        # value 0.5 → bin [0.33, 0.66): rows 2
        (0.5, ["c"]),
        # value 0.7 → bin [0.66, 1]: rows 3 and 4
        (0.7, ["d", "e"]),
    ],
)
def test_filter_dataframe_to_binned_value_1d(binned_dataframe, value, expected_labels):
    bin_edges = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    result = filter_dataframe_to_binned_value(binned_dataframe, "feat", value, bin_edges)
    assert result["label"].tolist() == expected_labels


def test_filter_dataframe_to_binned_value_1d_list_args(binned_dataframe):
    """Passing lists instead of scalars/arrays for a 1-D case should work identically."""
    bin_edges = np.array([0.0, 0.5, 1.0])
    result_scalar = filter_dataframe_to_binned_value(binned_dataframe, "feat", 0.2, bin_edges)
    result_list = filter_dataframe_to_binned_value(binned_dataframe, ["feat"], [0.2], [bin_edges])
    pd.testing.assert_frame_equal(
        result_scalar.reset_index(drop=True), result_list.reset_index(drop=True)
    )


def test_filter_dataframe_to_binned_value_2d():
    """Filtering in 2-D feature space returns only rows whose bin matches in both dimensions."""
    df = pd.DataFrame(
        {
            "dim_1": [0.1, 0.1, 0.7, 0.7],
            "dim_2": [0.2, 0.8, 0.2, 0.8],
            "label": ["a", "b", "c", "d"],
        }
    )
    bin_edges_1 = np.array([0.0, 0.5, 1.0])
    bin_edges_2 = np.array([0.0, 0.5, 1.0])

    # target: dim_1 ~ 0.1 (bin 0) AND dim_2 ~ 0.8 (bin 1) → only row "b"
    result = filter_dataframe_to_binned_value(
        df,
        columns=["dim_1", "dim_2"],
        values=[0.1, 0.8],
        bin_edges=[bin_edges_1, bin_edges_2],
    )
    assert result["label"].tolist() == ["b"]


def test_filter_dataframe_to_binned_value_value_at_upper_boundary_clamped_to_last_bin():
    """A target value exactly equal to the last bin edge maps to the last valid bin,
    and dataframe rows whose feature value equals the upper edge are also included.
    """
    df = pd.DataFrame({"feat": [0.4, 0.6, 0.8, 1.0], "label": ["a", "b", "c", "d"]})
    bin_edges = np.array([0.0, 0.5, 1.0])
    # target 1.0 → clamped to bin 1; rows with feat in (0.5, 1.0] all match bin 1
    result = filter_dataframe_to_binned_value(df, "feat", 1.0, bin_edges)
    assert result["label"].tolist() == ["b", "c", "d"]


def test_filter_dataframe_to_binned_value_value_below_lower_boundary_raises():
    """A target value below the lower bin edge raises a ValueError."""
    df = pd.DataFrame({"feat": [0.1, 0.3, 0.6], "label": ["a", "b", "c"]})
    bin_edges = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="outside the range of bin edges"):
        filter_dataframe_to_binned_value(df, "feat", -0.5, bin_edges)


def test_filter_dataframe_to_binned_value_value_above_upper_boundary_raises():
    """A target value above the upper bin edge raises a ValueError."""
    df = pd.DataFrame({"feat": [0.1, 0.3, 0.6], "label": ["a", "b", "c"]})
    bin_edges = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="outside the range of bin edges"):
        filter_dataframe_to_binned_value(df, "feat", 1.5, bin_edges)


def test_filter_dataframe_to_binned_value_no_matching_rows_returns_empty():
    """When no rows fall in the target bin, an empty (but schema-correct) DataFrame is returned."""
    df = pd.DataFrame({"feat": [0.1, 0.2, 0.3], "label": ["a", "b", "c"]})
    bin_edges = np.array([0.0, 0.5, 1.0])
    # all feat values are in bin 0; targeting bin 1 yields no rows
    result = filter_dataframe_to_binned_value(df, "feat", 0.7, bin_edges)
    assert result.empty
    assert list(result.columns) == list(df.columns)


def test_filter_dataframe_to_binned_value_preserves_original_index(binned_dataframe):
    """The returned dataframe preserves the original row index (no implicit reset)."""
    bin_edges = np.array([0.0, 0.5, 1.0])
    result = filter_dataframe_to_binned_value(binned_dataframe, "feat", 0.6, bin_edges)
    # rows 2, 3, 4 have feat values 0.6, 0.8, 0.9 → all in bin 1
    assert result.index.tolist() == [2, 3, 4]


@pytest.mark.parametrize(
    "columns, values, bin_edges",
    [
        # 2 columns but only 1 value
        (["dim_1", "dim_2"], [0.1], [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]),
        # 1 column but 2 bin_edges arrays
        (["dim_1"], [0.1, 0.2], [np.array([0.0, 0.5, 1.0])]),
        # 2 columns but 0 bin_edges arrays
        (["dim_1", "dim_2"], [0.1, 0.2], []),
    ],
)
def test_filter_dataframe_to_binned_value_raises_on_length_mismatch(columns, values, bin_edges):
    df = pd.DataFrame({"dim_1": [0.1, 0.4], "dim_2": [0.6, 0.9]})
    with pytest.raises(ValueError, match="Length of columns, value, and bin_edges"):
        filter_dataframe_to_binned_value(df, columns, values, bin_edges)


@pytest.fixture
def shear_stress_dataframe():
    """Dataframe with three distinct shear stress values and an extra column."""
    return pd.DataFrame(
        {
            Column.SHEAR_STRESS: [1.0, 1.0, 5.0, 5.0, 10.0],
            "label": ["a", "b", "c", "d", "e"],
        }
    )


def test_filter_dataframe_by_shear_stress_returns_matching_rows(shear_stress_dataframe):
    """Only rows whose shear-stress value equals the requested value are returned."""
    result = filter_dataframe_by_shear_stress(shear_stress_dataframe, 5.0)
    assert result[Column.SHEAR_STRESS].tolist() == [5.0, 5.0]
    assert result["label"].tolist() == ["c", "d"]


def test_filter_dataframe_by_shear_stress_no_matching_rows_returns_empty(shear_stress_dataframe):
    """When no rows match the shear stress, an empty but schema-correct DataFrame is returned."""
    result = filter_dataframe_by_shear_stress(shear_stress_dataframe, 99.0)
    assert result.empty
    assert list(result.columns) == list(shear_stress_dataframe.columns)


def test_filter_dataframe_by_shear_stress_missing_column_raises():
    """A DataFrame that lacks Column.SHEAR_STRESS should raise a ValueError."""
    df = pd.DataFrame({"other_column": [1.0, 2.0]})
    with pytest.raises(ValueError, match="DataFrame must contain column"):
        filter_dataframe_by_shear_stress(df, 1.0)


def test_filter_dataframe_by_shear_stress_returns_copy(shear_stress_dataframe):
    """Mutating the returned DataFrame must not affect the original."""
    result = filter_dataframe_by_shear_stress(shear_stress_dataframe, 1.0)
    result[Column.SHEAR_STRESS] = -1.0
    assert shear_stress_dataframe[Column.SHEAR_STRESS].tolist() == [1.0, 1.0, 5.0, 5.0, 10.0]


@pytest.fixture()
def stability_dataframe():
    """DataFrame with rows covering all StabilityLabel values."""
    return pd.DataFrame(
        {
            Column.FIXED_POINT_STABILITY: [
                StabilityLabel.STABLE,
                StabilityLabel.SADDLE,
                StabilityLabel.UNSTABLE,
                StabilityLabel.INDETERMINATE,
                StabilityLabel.NODE,
                StabilityLabel.SPIRAL,
                StabilityLabel.STABLE,
            ],
            "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
    )


def test_filter_dataframe_by_stability_returns_matching_rows(stability_dataframe):
    """Rows whose stability column matches the requested label are returned."""
    result = filter_dataframe_by_stability(stability_dataframe, StabilityLabel.STABLE)
    assert list(result[Column.FIXED_POINT_STABILITY]) == [
        StabilityLabel.STABLE,
        StabilityLabel.STABLE,
    ]
    assert result["x"].tolist() == [0.1, 0.7]


def test_filter_dataframe_by_stability_each_label(stability_dataframe):
    """Filtering by each valid StabilityLabel returns only rows with that label."""
    for label in StabilityLabel:
        result = filter_dataframe_by_stability(stability_dataframe, label)
        assert all(result[Column.FIXED_POINT_STABILITY] == label)


def test_filter_dataframe_by_stability_no_matching_rows_returns_empty(stability_dataframe):
    """When no rows have the requested label, an empty DataFrame with all columns is returned."""
    df_no_node = stability_dataframe[
        stability_dataframe[Column.FIXED_POINT_STABILITY] != StabilityLabel.NODE
    ].reset_index(drop=True)
    result = filter_dataframe_by_stability(df_no_node, StabilityLabel.NODE)
    assert result.empty
    assert list(result.columns) == list(df_no_node.columns)


def test_filter_dataframe_by_stability_preserves_original_index(stability_dataframe):
    """The returned DataFrame preserves the original row indices (no implicit reset)."""
    result = filter_dataframe_by_stability(stability_dataframe, StabilityLabel.STABLE)
    assert result.index.tolist() == [0, 6]


def test_filter_dataframe_by_stability_missing_column_raises():
    """A ValueError is raised when the stability column is absent from the DataFrame."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        filter_dataframe_by_stability(df, StabilityLabel.STABLE)


def test_filter_dataframe_by_stability_does_not_mutate_input(stability_dataframe):
    """The original DataFrame is not modified by the filtering operation."""
    original_len = len(stability_dataframe)
    filter_dataframe_by_stability(stability_dataframe, StabilityLabel.SADDLE)
    assert len(stability_dataframe) == original_len
