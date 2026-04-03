"""Methods for computing forward differences of time series data."""

import logging
from typing import overload

import numpy as np
import pandas as pd

from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import PERIOD_THETA_RESCALED

logger = logging.getLogger(__name__)


@overload
def _take_dataframe_column_diff(
    dataframe_column: pd.Series, diff_step: int, fillna_value: float | None = None
) -> pd.Series: ...


@overload
def _take_dataframe_column_diff(
    dataframe_column: pd.DataFrame, diff_step: int, fillna_value: float | None = None
) -> pd.DataFrame: ...


def _take_dataframe_column_diff(
    dataframe_column: pd.Series | pd.DataFrame, diff_step: int, fillna_value: float | None = None
) -> pd.Series | pd.DataFrame:
    """
    Helper function to take the difference along a columns of a DataFrame, given
    a specified step size.

    The returned Series or DataFrame will contain the differences along the
    input column(s), with NaN values at the end where the difference could not
    be computed due to shifting. If a fillna_value is provided, NaN values will
    be replaced with the fillna_value.

    Parameters
    ----------
    dataframe_column
        A column of a DataFrame.
    diff_step
        The number of rows ahead to take the difference with.
    fillna_value
        Optional, value to fill NaN values with after taking the difference.
    """
    diffed_column = dataframe_column.diff(periods=diff_step).shift(-diff_step)
    if fillna_value is not None:
        diffed_column = diffed_column.fillna(fillna_value)
    return diffed_column


def compute_forward_differences_along_trajectory(
    df_traj: pd.DataFrame,
    column_names: list,
    polar_angle_period: float = PERIOD_THETA_RESCALED,
    time_lag: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute forward differences at a given time lag along a trajectory in
    feature space.

    **Polar angle handling**

    If one of the input ``column_names`` is 'polar_theta', the function will
    compute circular differences for the polar angle feature using the given
    ``polar_angle_period``. Specifically, it will unwrap the polar angle
    trajectory according to the given period for each crop before computing
    differences.

    **Time lag handling**

    The input ``time_lag`` determines the time lag (in number of frames) to use
    when computing forward differences. By default, it is set to 1, which
    corresponds to forward differences between consecutive timepoints. If a
    different time lag is specified, the function will compute differences
    between timepoints that are separated by ``time_lag`` number of frames.

    Parameters
    ----------
    df_traj
        DataFrame containing the feature data for one crop trajectory.
    column_names
        List of column names corresponding to the features of interest in the
        DataFrame.
    polar_angle_period
        Period of the polar angle feature, used to compute circular differences
        for angular data.
    time_lag
        Time lag (in number of frames) for forward difference calculation.

    Returns
    -------
    :
        Array of feature values along the trajectory for the specified columns.
    :
        Array of forward differences in feature values along the trajectory for
        the specified columns.
    """
    # initialize name for difference columns
    diff_column_names = [f"{col}{Column.DiffAEData.DIFFERENCE_SUFFIX}" for col in column_names]
    timepoint_diff_column = f"{Column.TIMEPOINT}{Column.DiffAEData.DIFFERENCE_SUFFIX}"

    # add column giving difference in timepoint between rows separated by
    # time_lag convert NaN to 0 -- occurs at end of trajectory
    df_traj[timepoint_diff_column] = _take_dataframe_column_diff(
        df_traj[Column.TIMEPOINT], time_lag, fillna_value=0
    )

    # add columns giving difference in feature values between consecutive
    # dataframe rows
    df_traj[diff_column_names] = _take_dataframe_column_diff(
        df_traj[column_names], time_lag, fillna_value=0
    )

    # if one of the column names is `polar_theta`, need to replace with the
    # circular difference for angular data instead of simple difference
    if Column.DiffAEData.POLAR_ANGLE in column_names:
        angle_diff_column = f"{Column.DiffAEData.POLAR_ANGLE}{Column.DiffAEData.DIFFERENCE_SUFFIX}"
        df_traj[f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped"] = np.unwrap(
            df_traj[Column.DiffAEData.POLAR_ANGLE].values, period=polar_angle_period
        )
        df_traj[angle_diff_column] = _take_dataframe_column_diff(
            df_traj[f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped"], time_lag, fillna_value=0
        )
        df_traj.drop(columns=[f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped"], inplace=True)

    # trajectory values to keep -- only keep steps where time difference is <=
    # time_lag which includes the last point in the trajectory (which has time
    # difference set to 0)
    traj_mask = df_traj[timepoint_diff_column] <= time_lag
    filtered_traj_array = df_traj[traj_mask][column_names].to_numpy()
    if time_lag > 1:
        # drop last time_lag - 1 points, as there is no valid difference there
        filtered_traj_array = filtered_traj_array[: -time_lag + 1]

    # for the gradient, only keep steps where time difference is exactly
    # time_lag frames i.e., no valid difference at the end of the trajectory
    # (only forward differences)
    gradient_mask = df_traj[timepoint_diff_column] == time_lag
    filtered_d_traj_array = df_traj[gradient_mask][diff_column_names].to_numpy()

    return filtered_traj_array, filtered_d_traj_array


def get_traj_and_diff(
    df: pd.DataFrame,
    column_names: list,
    polar_angle_period: float = PERIOD_THETA_RESCALED,
    time_lag: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Get trajectories and single-timepoint displacement vectors (forward
    differences) for each single-crop trajectory in feature space.

    **Input dataframe**

    The input dataframe should have columns for:
        - frame_number: timepoint of the crop
        - crop_index: unique index for each crop
        - columns for each feature (e.g., pc_0, pc_1, pc_2, ...)
           matching input ``column_names``

    See documentation for `compute_forward_differences_along_trajectory` for
    more details on the numerical calculation of the forward differences.

    Parameters
    ----------
    df
        DataFrame with columns for each feature.
    column_names
        List of column names corresponding to the features of interest in the
        DataFrame.
    polar_angle_period
        Period of the polar angle feature, used to compute circular differences
        for angular data.
    time_lag
        Time lag (in number of frames) for forward difference calculation.

    Returns
    -------
    :
        List of individual crop trajectories in feature space.
    :
        List of displacement vectors along each trajectory in feature space.
    """
    # check that required columns are present
    required_columns = [Column.TIMEPOINT, Column.CROP_INDEX, *column_names]
    check_required_columns_in_dataframe(df, required_columns)

    # initialize lists for storing outputs
    traj_list = []
    d_traj_list = []

    # loop over each crop in the dataset
    for _, df_crop in df.groupby(Column.CROP_INDEX):
        # skip if time_lag is larger than number of timepoints in this trajectory
        if time_lag > df_crop[Column.TIMEPOINT].nunique():
            continue

        # sort by timepoint to ensure that trajectory is in correct order before
        # computing differences
        df_crop_ = df_crop.sort_values(by=Column.TIMEPOINT)

        # compute forward differences along trajectory for this crop, and filter
        # to keep only differences between timepoints that are separated by
        # time_lag number of frames (accounts for any missing timepoints in the
        # trajectory, for example due to outlier filtering)
        filtered_traj, filtered_d_traj = compute_forward_differences_along_trajectory(
            df_crop_, column_names, polar_angle_period, time_lag
        )

        # if either the returned trajectory or difference arrays are empty, skip
        # this trajectory
        if filtered_traj.size == 0 or filtered_d_traj.size == 0:
            continue

        # else, append and continue through the loop
        traj_list.append(filtered_traj)
        d_traj_list.append(filtered_d_traj)

    # if lists are empty, log warning
    if len(traj_list) == 0 or len(d_traj_list) == 0:
        logger.warning(
            "No valid trajectories found after computing forward differences with time lag [ %s ]. "
            "Check that the input dataframe has the required columns and that the time lag is not "
            "larger than the number of timepoints in the trajectories.",
            time_lag,
        )
    return traj_list, d_traj_list
