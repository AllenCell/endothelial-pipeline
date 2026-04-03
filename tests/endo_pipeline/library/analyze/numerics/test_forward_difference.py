import numpy as np
import pandas as pd
import pytest

from endo_pipeline.library.analyze.numerics.forward_difference import (
    compute_forward_differences_along_trajectory,
    get_traj_and_diff,
)
from endo_pipeline.settings.column_names import ColumnName as Column


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
