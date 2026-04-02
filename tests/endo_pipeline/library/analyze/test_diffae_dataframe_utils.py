import numpy as np
import pandas as pd
import pytest

from endo_pipeline.configs import (
    ChannelIndices,
    DatasetConfig,
    PositionAnnotation,
    TimepointAnnotation,
)
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    compute_forward_differences_along_trajectory,
    filter_dataframe_by_annotations,
    filter_dataframe_by_track_length,
    get_latent_feature_column_names_from_dataframe,
    get_traj_and_diff,
    project_features_to_pcs,
    unwrap_nonsequential_array,
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


@pytest.fixture
def dataframe():
    timepoints = list(range(4))
    positions = [1, 3, 5]
    positions_tiled = [f"P{i}" for i in positions for _ in timepoints]
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
            ["P3"] * 2 + ["P5"] * 3,
            [2, 3, 1, 2, 3],
        ),
        (
            PositionAnnotation.DUST_ARTIFACT,
            [TimepointAnnotation.CELL_PILING],
            ["P3"] * 2 + ["P5"] * 3,
            [0, 1, 0, 1, 2],
        ),
        (
            [],
            [TimepointAnnotation.CELL_PILING],
            ["P1"] * 4 + ["P3"] * 2 + ["P5"] * 3,
            [0, 1, 2, 3, 0, 1, 0, 1, 2],
        ),
        (
            [],
            [TimepointAnnotation.NOT_STEADY_STATE],
            ["P1"] * 2 + ["P3"] * 2 + ["P5"] * 3,
            [2, 3, 2, 3, 1, 2, 3],
        ),
        (
            [],
            [TimepointAnnotation.AUTO_BF_SCOPE_ERROR],
            ["P1"] * 3 + ["P3"] * 4 + ["P5"] * 4,
            [0, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ),
        ([], None, ["P1"] * 2 + ["P5"] * 2, [2, 3, 1, 2]),
        ([PositionAnnotation.DUST_ARTIFACT], [], ["P3"] * 4 + ["P5"] * 4, [0, 1, 2, 3] * 2),
        (None, None, ["P5", "P5"], [1, 2]),
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
    "dataframe, expected_column_names",
    [
        (
            pd.DataFrame(
                {f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}": [0.1 * i] * 5 for i in range(10)}
            ),
            [f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}" for i in range(10)],
        ),
        (
            pd.DataFrame(
                {
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}_suffix": [0.2 * i] * 3
                    for i in range(10)
                }
            ),
            [],
        ),
        (
            pd.DataFrame(
                {
                    "other_column": [1, 2, 3],
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}0": [0.0, 0.0, 0.0],
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}1": [0.1, 0.1, 0.1],
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}0_extra": [0.2, 0.2, 0.2],
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}one": [0.2, 0.2, 0.2],
                }
            ),
            [
                f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}0",
                f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}1",
            ],
        ),
    ],
)
def test_get_latent_feature_column_names_from_dataframe(dataframe, expected_column_names):
    latent_feature_columns = get_latent_feature_column_names_from_dataframe(dataframe)
    assert latent_feature_columns == expected_column_names


@pytest.mark.parametrize(
    "dataframe, column_names, expected_trajectories, expected_differences",
    [
        (
            pd.DataFrame(
                {
                    f"{Column.TIMEPOINT}": [0, 1, 2] * 3,
                    f"{Column.CROP_INDEX}": [0, 0, 0] + [1, 1, 1] + [2, 2, 2],
                    f"{Column.DiffAEData.POLAR_ANGLE}": [np.pi, np.pi - 0.05, -np.pi + 0.05]
                    + [0.1, -0.4, 0.3]
                    + [1.0, 1.1, 1.2],
                    "pc_3": [2.5, 2.6, 2.7] + [0.1, 0.55, 0.2] + [1.5, 1.6, 1.7],
                }
            ),
            [f"{Column.DiffAEData.POLAR_ANGLE}", "pc_3"],
            [
                np.array(
                    [
                        [np.pi, 2.5],
                        [np.pi - 0.05, 2.6],
                        [-np.pi + 0.05, 2.7],
                    ]
                ),  # crop 0
                np.array(
                    [
                        [0.1, 0.1],
                        [-0.4, 0.55],
                        [0.3, 0.2],
                    ]
                ),  # crop 1
                np.array(
                    [
                        [1.0, 1.5],
                        [1.1, 1.6],
                        [1.2, 1.7],
                    ]
                ),  # crop 2
            ],
            [
                np.array(
                    [
                        [-0.05, 0.1],
                        [0.1, 0.1],
                    ]
                ),  # crop 0
                np.array(
                    [
                        [-0.5, 0.45],
                        [0.7, -0.35],
                    ]
                ),  # crop 1
                np.array(
                    [
                        [0.1, 0.1],
                        [0.1, 0.1],
                    ]
                ),  # crop 2
            ],
        ),
        (  # test dropping non-consecutive timepoints
            pd.DataFrame(
                {
                    f"{Column.TIMEPOINT}": [0, 2, 3] * 3,
                    f"{Column.CROP_INDEX}": [0, 0, 0] + [1, 1, 1] + [2, 2, 2],
                    f"{Column.DiffAEData.POLAR_RADIUS}": [0.1, 0.75, 0.6]
                    + [1.0, 1.5, 1.24]
                    + [2.5, 3.0, 3.2],
                    "pc_3": [2.5, 2.7, 2.5] + [0.1, 0.2, 0.3] + [1.5, 1.7, 1.95],
                }
            ),
            [f"{Column.DiffAEData.POLAR_RADIUS}", "pc_3"],
            [
                np.array(
                    [
                        [0.75, 2.7],
                        [0.6, 2.5],
                    ]
                ),
                np.array(
                    [
                        [1.5, 0.2],
                        [1.24, 0.3],
                    ]
                ),
                np.array(
                    [
                        [3.0, 1.7],
                        [3.2, 1.95],
                    ]
                ),
            ],
            [
                np.array(
                    [
                        [-0.15, -0.2],
                    ]
                ),
                np.array(
                    [
                        [-0.26, 0.1],
                    ]
                ),
                np.array(
                    [
                        [0.2, 0.25],
                    ]
                ),
            ],
        ),
    ],
)
def test_get_traj_and_diff(dataframe, column_names, expected_trajectories, expected_differences):
    trajectories, differences = get_traj_and_diff(dataframe, column_names)

    # assert expected lengths of the returned lists
    assert len(trajectories) == len(expected_trajectories)
    assert len(differences) == len(expected_differences)
    assert len(trajectories) == len(differences)

    # make sure returned trajectory and difference arrays have expected shapes
    n_dim = len(column_names)
    timepoint_diff_column = f"{Column.TIMEPOINT}{Column.DiffAEData.DIFFERENCE_SUFFIX}"
    for crop_index, df_crop in dataframe.groupby(Column.CROP_INDEX):
        traj: np.ndarray = trajectories[crop_index]
        diff: np.ndarray = differences[crop_index]
        # check that timepoint differences > 1 are dropped
        df_crop[timepoint_diff_column] = df_crop[Column.TIMEPOINT].diff().shift(-1).fillna(0)
        valid_timepoints = df_crop[timepoint_diff_column] <= 1
        n_valid_frames = len(df_crop[valid_timepoints])
        assert traj.shape == (n_valid_frames, n_dim)
        assert diff.shape == (n_valid_frames - 1, n_dim)

    # assert array values are almost equal
    for traj, expected_traj in zip(trajectories, expected_trajectories, strict=True):
        np.testing.assert_array_almost_equal(traj, expected_traj)
    for diff, expected_diff in zip(differences, expected_differences, strict=True):
        np.testing.assert_array_almost_equal(diff, expected_diff)


@pytest.mark.parametrize(
    "wrapped_array, period, expected_unwrapped_array",
    [
        (
            np.array([0.0, np.pi / 2, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 2 * np.pi]),
            2 * np.pi,
            np.array([0.0, np.pi / 2, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0.0]),
        ),
        (
            np.array([1.0, 1.5, -2.5, -2.0, 2.0, 2.5]),
            5.0,
            np.array([1.0, 1.5, 2.5, 3.0, 2.0, 2.5]),
        ),
        (
            np.array([10.0, 12.0, 13.0, 9.0, 16.0]),
            5.0,
            np.array([10.0, 12.0, 8.0, 9.0, 11.0]),
        ),
    ],
)
def test_unwrap_nonsequential_array(wrapped_array, period, expected_unwrapped_array):
    unwrapped_array = unwrap_nonsequential_array(wrapped_array, period)
    np.testing.assert_array_almost_equal(unwrapped_array, expected_unwrapped_array)


@pytest.mark.parametrize(
    "num_components, provide_feature_columns, compute_polar, flip_pc3_sign, raises_error",
    [
        (  # fit PCA with 3 components, project to 3 PCs, don't compute polar angle, don't flip PC3 sign
            3,
            True,
            False,
            False,
            False,  # should not raise error since not computing polar angle or flipping PC3 sign
        ),
        (  # confirm that passing None for feature_columns runs as expected (i.e., gets them from the dataframe)
            3,
            False,
            False,
            False,
            False,
        ),
        (  # fit PCA with 2 components, project to 2 PCs, compute polar angle, don't flip PC3 sign
            2,
            True,
            True,
            False,
            False,  # should not raise error since using 2 PCs to compute polar angle and not flipping PC3 sign
        ),
        (  # check that error is raised if trying to compute polar angle with only 1 PC
            1,
            True,
            True,
            False,
            True,
        ),
        (  # check that error is raised if trying to flip PC3 sign when only 2 PCs are computed
            2,
            True,
            True,
            True,
            True,
        ),
    ],
)
def test_project_features_to_pcs(
    num_components, provide_feature_columns, compute_polar, flip_pc3_sign, raises_error
):
    from sklearn.decomposition import PCA

    # create a simple test dataframe with 3 latent feature columns
    df = pd.DataFrame(
        {
            "feat_0": [1.0, 2.0, 3.0],
            "feat_1": [4.0, 5.0, 6.0],
            "feat_2": [7.0, 8.0, 9.0],
        }
    )

    if provide_feature_columns:
        feature_columns = [f"feat_{i}" for i in range(3)]
    else:
        feature_columns = None

    pca_model = PCA(n_components=num_components).fit(df.values)

    # project to PCs, check that the function runs without error when expected
    # and raises error when expected
    try:
        projected_df = project_features_to_pcs(
            df,
            pca_model,
            feat_cols=feature_columns,
            compute_polar=compute_polar,
            flip_pc3_sign=flip_pc3_sign,
        )

        # check that the projected dataframe has the expected columns
        # should still have original columns
        expected_projected_df_columns = df.columns.tolist()
        # should have new columns for each projected PC
        # (convention is to name them with PCA_FEATURE_PREFIX followed
        # by the PC number starting from 1, not 0)
        expected_projected_df_columns.extend([f"pc_{i+1}" for i in range(num_components)])
        # if computing polar angle, should have new columns for polar angle and radius
        if compute_polar:
            expected_projected_df_columns.append(Column.DiffAEData.POLAR_ANGLE)
            expected_projected_df_columns.append(Column.DiffAEData.POLAR_RADIUS)
        # if flipping PC3 sign, should have new column for flipped PC3 value
        if flip_pc3_sign:
            expected_projected_df_columns.append(f"{Column.DiffAEData.PC3_FLIPPED}")
        assert set(projected_df.columns) == set(expected_projected_df_columns)

        # check that the PCA model has the expected number of components
        assert pca_model.n_components_ == num_components
    except ValueError:  # if a ValueError is raised, check that it was expected
        assert raises_error


@pytest.mark.parametrize(
    "dataframe, track_length_column, minimum_track_length, expected_filtered_dataframe",
    [
        (
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [1, 2, 3, 4, 5],
                    "other_column": [10, 20, 30, 40, 50],
                }
            ),
            Column.TRACK_LENGTH,
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
            Column.TRACK_LENGTH,
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
    dataframe, track_length_column, minimum_track_length, expected_filtered_dataframe
):
    filtered_df = filter_dataframe_by_track_length(
        dataframe, track_length_column, minimum_track_length
    )
    pd.testing.assert_frame_equal(filtered_df, expected_filtered_dataframe, check_like=True)


@pytest.mark.parametrize(
    "dataframe, track_length_column, minimum_track_length",
    [
        (
            pd.DataFrame(
                {
                    "some_other_column": [1, 2, 3],
                }
            ),
            Column.TRACK_LENGTH,
            3,
        ),
        (
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [1, 2, 3],
                }
            ),
            "non_existent_column",
            3,
        ),
    ],
)
def test_filter_dataframe_by_track_length_invalid_column(
    dataframe, track_length_column, minimum_track_length
):
    with pytest.raises(ValueError):
        filter_dataframe_by_track_length(dataframe, track_length_column, minimum_track_length)


def test_filter_dataframe_by_track_length_all_filtered_out():
    # if all tracks are shorter than minimum_track_length, should return empty dataframe with same columns
    dataframe = pd.DataFrame(
        {
            Column.TRACK_LENGTH: [1, 2, 3],
            "other_column": [10, 20, 30],
        }
    )
    with pytest.raises(ValueError):
        filter_dataframe_by_track_length(dataframe, Column.TRACK_LENGTH, 4)


@pytest.mark.parametrize(
    "timepoints, feature_values, time_lag, expected_traj, expected_d_traj",
    [
        (
            # Basic case: consecutive timepoints, time_lag=1
            # Last point has diff=0 so it's in traj but not d_traj
            [0, 1, 2, 3],
            [1.0, 2.0, 4.0, 7.0],
            1,
            np.array([[1.0], [2.0], [4.0], [7.0]]),
            np.array([[1.0], [2.0], [3.0]]),
        ),
        (
            # Non-consecutive timepoints: gap of 2 between t=0 and t=2
            # t=0 has timepoint diff=2 > time_lag=1, so excluded from both traj and d_traj
            # t=2 has diff=1, included in traj and d_traj
            # t=3 has diff=0 (fillna), included in traj but not d_traj
            [0, 2, 3],
            [1.0, 2.0, 3.0],
            1,
            np.array([[2.0], [3.0]]),
            np.array([[1.0]]),
        ),
        (
            # time_lag=2: consecutive timepoints [0, 1, 2, 3]
            # All timepoints initially pass traj_mask (timepoint diffs: 2, 2, 0, 0; all <= 2)
            # Last time_lag-1=1 point is then dropped from traj
            # Only first two in d_traj (timepoint diffs == 2)
            [0, 1, 2, 3],
            [1.0, 2.0, 4.0, 7.0],
            2,
            np.array([[1.0], [2.0], [4.0]]),
            np.array([[3.0], [5.0]]),
        ),
        (
            # time_lag=2 with a gap: timepoints [0, 1, 3]
            # t=0: timepoint diff=3 > 2, excluded by traj_mask
            # t=1: diff=0 (fillna), passes traj_mask
            # t=3: diff=0 (fillna), passes traj_mask
            # After dropping last time_lag-1=1 point: only t=1 remains in traj
            # No pairs separated by exactly 2 frames, so d_traj is empty
            [0, 1, 3],
            [1.0, 2.0, 4.0],
            2,
            np.array([[2.0]]),
            np.zeros((0, 1)),
        ),
    ],
)
def test_compute_forward_differences_along_trajectory_scalar_feature(
    timepoints, feature_values, time_lag, expected_traj, expected_d_traj
):
    col = "column_0"
    df_traj = pd.DataFrame(
        {
            Column.TIMEPOINT: timepoints,
            col: feature_values,
        }
    )
    traj, d_traj = compute_forward_differences_along_trajectory(df_traj, [col], time_lag=time_lag)
    np.testing.assert_array_almost_equal(traj, expected_traj)
    np.testing.assert_array_almost_equal(d_traj, expected_d_traj)


def test_compute_forward_differences_along_trajectory_multiple_features():
    """Multiple features are all returned in the correct column order."""
    cols = [f"column_{i}" for i in range(3)]
    df_traj = pd.DataFrame(
        {
            Column.TIMEPOINT: [0, 1, 2],
            cols[0]: [1.0, 2.0, 3.0],
            cols[1]: [10.0, 20.0, 30.0],
            cols[2]: [100.0, 200.0, 300.0],
        }
    )
    traj, d_traj = compute_forward_differences_along_trajectory(df_traj, cols, time_lag=1)

    # shapes: 3 timepoints in traj, 2 forward differences, 3 features
    assert traj.shape == (3, 3)
    assert d_traj.shape == (2, 3)

    np.testing.assert_array_almost_equal(
        traj,
        np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]]),
    )
    np.testing.assert_array_almost_equal(
        d_traj,
        np.array([[1.0, 10.0, 100.0], [1.0, 10.0, 100.0]]),
    )


def test_compute_forward_differences_along_trajectory_polar_angle_unwrapping():
    """
    When 'polar_theta' is in column_names, differences are computed using
    np.unwrap so that a wrap-around near the period boundary produces a small
    difference instead of a large jump.
    """
    period = np.pi  # matches PERIOD_THETA_RESCALED
    eps = 0.05
    # angles [pi/2 - eps, -(pi/2 - eps), 0.0]:
    # without unwrapping: diff[0] = -(pi - 2*eps)  (large negative)
    # with unwrapping via period=pi: diff[0] = 2*eps  (small positive)
    angles = [np.pi / 2 - eps, -(np.pi / 2 - eps), 0.0]
    df_traj = pd.DataFrame(
        {
            Column.TIMEPOINT: [0, 1, 2],
            Column.DiffAEData.POLAR_ANGLE: angles,
        }
    )
    traj, d_traj = compute_forward_differences_along_trajectory(
        df_traj,
        [Column.DiffAEData.POLAR_ANGLE.value],
        polar_angle_period=period,
        time_lag=1,
    )

    # Trajectory values are the raw (non-unwrapped) angles
    np.testing.assert_array_almost_equal(traj[:, 0], angles)

    # Differences should be based on the unwrapped sequence
    unwrapped = np.unwrap(np.array(angles), period=period)
    expected_diffs = np.diff(unwrapped)
    np.testing.assert_array_almost_equal(d_traj[:, 0], expected_diffs)
    # Specifically: the wrap-around diff should be small (2*eps), not large
    assert abs(d_traj[0, 0]) < np.pi / 2


def test_compute_forward_differences_along_trajectory_single_timepoint():
    """A single-row trajectory produces an empty differences array."""
    col = "column_0"
    df_traj = pd.DataFrame(
        {
            Column.TIMEPOINT: [0],
            col: [5.0],
        }
    )
    traj, d_traj = compute_forward_differences_along_trajectory(df_traj, [col], time_lag=1)

    # Trajectory has 1 point; no forward differences possible
    assert traj.shape == (1, 1)
    assert d_traj.shape == (0, 1)
    np.testing.assert_array_almost_equal(traj, np.array([[5.0]]))
