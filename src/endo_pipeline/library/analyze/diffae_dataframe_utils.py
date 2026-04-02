import logging
import re
from typing import cast, overload

import numpy as np
import pandas as pd

from endo_pipeline.configs import (
    DatasetConfig,
    PositionAnnotation,
    TimepointAnnotation,
    get_all_unannotated_timepoints,
    get_frame_after_flow_change,
    get_unannotated_positions,
    load_dataset_config,
)
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import PERIOD_THETA_RESCALED

logger = logging.getLogger(__name__)


def filter_dataframe_by_track_length(
    dataframe: pd.DataFrame, track_length_column: str, minimum_track_length: int
) -> pd.DataFrame:
    """
    Filter dataframe to only include tracks above a minimum track length.

    **Error handling**

    If no tracks remain after filtering, a ValueError is raised with a message
    indicating that no tracks with length >= minimum_track_length remain after
    filtering, and suggesting to check the track length distribution and/or
    adjust the minimum_track_length threshold.

    Parameters
    ----------
    dataframe
        DataFrame containing data of interest, which must include a column for
        track length.
    track_length_column
        Name of the column containing track length values.
    minimum_track_length
        Minimum track length to filter tracks.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only tracks with length >=
        minimum_track_length.
    """

    logger.debug(
        "Filtering dataframe to only include tracks with length >= [ %s ] timepoints.",
        minimum_track_length,
    )
    logger.debug("Dataframe length before filtering: [ %s ] rows.", len(dataframe))
    # check that required columns are present in dataframe
    check_required_columns_in_dataframe(dataframe, [track_length_column])
    dataframe_filtered = dataframe[dataframe[track_length_column] >= minimum_track_length]

    # if empty dataframe after filtering, raise error
    if dataframe_filtered.empty:
        logger.error(
            "No tracks with length >= minimum_track_length [ %s ] after filtering. "
            "Check track length distribution and/or adjust minimum_track_length.",
            minimum_track_length,
        )
        raise ValueError(
            f"No tracks with length >= minimum_track_length [ {minimum_track_length} ] after filtering. "
            "Check track length distribution and/or adjust minimum_track_length."
        )

    # reset index of filtered dataframe
    dataframe_filtered = dataframe_filtered.reset_index(drop=True)

    logger.debug("Dataframe length after filtering: [ %s ] rows.", len(dataframe_filtered))

    return dataframe_filtered


def filter_dataframe_by_annotations(
    dataframe: pd.DataFrame,
    dataset_config: DatasetConfig,
    position_annotations: list[PositionAnnotation] | None = None,
    timepoint_annotations: list[TimepointAnnotation] | None = None,
) -> pd.DataFrame:
    """
    Remove annotated timepoints and positions from a dataframe of DiffAE features for one dataset.

    Default behavior is to remove all annotated timepoints and positions.

    Parameters
    ----------
    dataframe
        Dataframe of features for one dataset.
    dataset_config
        Dataset config for the dataset.
    position_annotations
        List of position annotations to remove. Use None to remove all annotated positions.
    timepoint_annotations
        List of timepoint annotations to remove. Use None to remove all annotated timepoints.

    Returns
    -------
    :
        Dataframe with annotated timepoints removed.
    """

    # check that required columns are present in dataframe
    required_columns = [Column.DATASET, Column.POSITION, Column.TIMEPOINT]
    check_required_columns_in_dataframe(dataframe, required_columns)

    if dataframe[Column.DATASET].nunique() != 1:
        logger.error("Dataframe must be restricted to one dataset only.")
        raise ValueError("Dataframe must be restricted to one dataset only.")

    if dataframe[Column.DATASET].unique()[0] != dataset_config.name:
        logger.error("Dataset name in dataframe does not match dataset name in dataset config.")
        raise ValueError("Dataset name in dataframe does not match dataset name in dataset config.")

    # get positions and timepoints to include based on annotations
    only_include_positions = get_unannotated_positions(dataset_config, position_annotations)
    only_include_positions_str = [f"P{pos}" for pos in only_include_positions]
    only_include_frames = get_all_unannotated_timepoints(dataset_config, timepoint_annotations)
    if dataframe[Column.POSITION].nunique() != len(dataset_config.zarr_positions):
        logger.warning("Expected dataframe to contain all positions in dataset, but it does not.")

    # filter dataframe to only include non-annotated positions
    # NOTE: temporary if-else until we update how we store position: replace 'P[int]' with int
    # this checks if all entries in the `.POSITION` column are strings that start with 'P'
    all_position_vals_start_with_P = (
        dataframe[Column.POSITION].transform(lambda pos: "P" in str(pos)).all()
    )
    if all_position_vals_start_with_P:
        position_type = "str"
        dataframe_exclude_positions = dataframe[
            dataframe[Column.POSITION].isin(only_include_positions_str)
        ]
    # otherwise it is assumed that the position column can be cast to `int`
    # (and if it can't be cast to `int`, an error will be raised later)
    else:
        position_type = "int"
        dataframe_exclude_positions = dataframe[
            dataframe[Column.POSITION].isin(only_include_positions)
        ]
    # filter dataframe to only include non-annotated timepoints
    df_filtered_list = []
    for position, df_position in dataframe_exclude_positions.groupby(Column.POSITION):
        # NOTE: temporary if-else until we update how we store position: replace 'P[int]' with int
        if position_type == "str":
            position_as_int = int(cast(str, position)[1:])
        else:
            position_as_int = cast(int, position)
        include_frames_for_position = only_include_frames.get(position_as_int, [])
        df_position_filtered = df_position[
            df_position[Column.TIMEPOINT].isin(include_frames_for_position)
        ]
        df_filtered_list.append(df_position_filtered)
    dataframe_filtered = pd.concat(df_filtered_list, ignore_index=True)

    return dataframe_filtered


def get_dataset_descriptions(
    list_of_datasets: list[str],
    include_duration: bool = True,
    simple: bool = False,
    include_shear_stress: bool = False,
) -> dict[str, str]:
    """
    Get descriptive metadata for each dataset given in the list of datasets.

    Describes the experimental conditions for each dataset,
        e.g., "48hr_Maximum_Shear_Stress_30_dyncm2".

    Parameters
    ----------
    list_of_datasets
        List of dataset names for which to get descriptions
    include_duration
        Include duration of each flow condition in description if true.
    simple
        Include description of shear regime (e.g., "High_Shear_Stress") if true.
    include_shear_stress
        Include exact shear stress value (e.g., "30_dyncm2") in description if true.

    Returns
    -------
    :
        A dictionary where keys are dataset names and values are descriptions.
    """

    description_dict = {}

    for dataset_name in list_of_datasets:
        config = load_dataset_config(dataset_name)
        description = []

        for condition, regime in zip(
            config.flow_conditions, config.shear_stress_regime, strict=True
        ):
            if include_duration:
                duration_in_frames = condition.stop - condition.start
                duration_in_hours = int(duration_in_frames * 5 / 60)
                description.append(f"{duration_in_hours}hr")

            if simple:
                description.append(regime.value)

            if not simple or include_shear_stress:
                description.append(f"{int(condition.shear_stress)}dyncm2")

        description_dict[dataset_name] = "_".join(description)

    return description_dict


def parse_dataset_description(dataset_description: str) -> str:
    """Parse dataset description for better readability in plot titles."""
    # replace underscores with spaces for better readability
    description_parsed = dataset_description.replace("_", " ")
    # find [0-9]dyncm2, put comma and space before, put a space between number and unit,
    # and change dyncm2 to dyn/cm^2 for better readability
    description_parsed = re.sub(r"(\d+)dyncm2", r", \1 dyn/cm$^2$", description_parsed)
    # turn capital 'S' into lowercase 's' for shear stress
    description_parsed = description_parsed.replace(" Shear Stress", " shear stress")
    # remove unwanted space before comma
    description_parsed = description_parsed.replace(" ,", ",")
    return description_parsed


def add_description_column(
    df: pd.DataFrame, dataset_name: str, simple: bool = False
) -> pd.DataFrame:
    """
    Add description column to DataFrame df.
    (Descriptions are currently based on the dataset name.).

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data for dataset dataset_name
        - IMPORTANT: DataFrame must be restricted to one dataset only,
            as identified by the dataset_name column
    - dataset_name: str, name of dataset to add description for
    - simple (optional): bool, whether to use simple description
        (e.g., "48hr_High")

    Outputs:
    - df: pd.DataFrame, DataFrame of feature data for one
        dataset with added description column
    """
    # get descriptions for each dataset name
    description = get_dataset_descriptions([dataset_name], simple=simple)

    # add description column to DataFrame
    df["description"] = description[dataset_name]  # add description to DataFrame

    return df


def df_to_array(df: pd.DataFrame, column_names: list) -> np.ndarray:
    """
    Convert DataFrame of features corresponding to one dataset to array
    of shape num_crops x num_timepoints x num_features.
    This function fills missing timepoints (for example filtered as outliers)
    with NaNs such that there is a row for every timepoint within the dataset
    duration for each crop.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data for one dataset
        - DataFrame should have metadata columns for crop_index and T
    - column_names: list[str], list of column names for features to include
        in output array

    Outputs:
    - feats: np.ndarray, array of feature data for all crops
        at all timepoints in one dataset
        - shape is num_crops x num_timepoints x num_features
    """
    # check that required columns are present in dataframe
    required_columns = [Column.CROP_INDEX, Column.TIMEPOINT, *column_names]
    check_required_columns_in_dataframe(df, required_columns)

    # get array of num crops x valid timepoints x num PCs, padding with NaNs
    # where timepoints are missing
    full_timepoint_range = (df[Column.TIMEPOINT].min(), df[Column.TIMEPOINT].max())

    feats = []
    for _, data_crop in df.groupby(Column.CROP_INDEX):
        data_crop = data_crop.sort_values(by=Column.TIMEPOINT)
        data_crop_filled = fill_missing_timepoints(data_crop, full_timepoint_range)
        feats.append(data_crop_filled[column_names].values)

    return np.array(feats)


def split_dataset_by_flow(
    df_proj: pd.DataFrame, dataset_config: DatasetConfig
) -> tuple[list[pd.DataFrame], list[float]]:
    """
    Get crop-based feature data (Diffusion AE output) for each flow condition present in a dataset.

    If there is only one flow condition, this method returns a lists of length 1
    containing the original dataframe and single shear stress value.

    Parameters
    ----------
    df_proj
        DataFrame containing the PCA-projected feature data for one dataset.
    dataset_config
        DatasetConfig object for the given dataset.

    Returns
    -------
    :
        List of DataFrames, each containing the feature data for one flow condition.
    :
        List of shear stress values for each flow condition.
    """
    # check that required columns are present
    check_required_columns_in_dataframe(df_proj, [Column.TIMEPOINT])

    # get flow condition information from dataset config
    flow_conditions = dataset_config.flow_conditions

    # split out data by flow condition,
    # starting with first flow condition
    first_shear = flow_conditions[0].shear_stress
    # initialize list of shear stress conditions
    shear_list = [first_shear]
    # if there is a change in flow condition
    if len(flow_conditions) > 1:
        # get frame number where second flow condition starts
        change_frame = get_frame_after_flow_change(dataset_config)
        # get second shear stress condition
        second_shear = flow_conditions[1].shear_stress
        shear_list.append(second_shear)
        logger.debug("Shear stress [ %s ] dyn/cm^2 until frame [ %s ]", first_shear, change_frame)
        logger.debug("Shear stress [ %s ] dyn/cm^2 after frame [ %s ]", second_shear, change_frame)
        # separate data into two dataframes based on
        # frame number where flow condition changes
        data_flow1 = df_proj[df_proj[Column.TIMEPOINT] < change_frame].copy()
        data_flow2 = df_proj[df_proj[Column.TIMEPOINT] >= change_frame].copy()
        # return list of dataframes for each flow condition
        data_all = [data_flow1, data_flow2]
    # else, there is only one flow condition
    else:
        logger.debug("Constant shear stress [ %s ] dyn/cm^2", first_shear)
        # list of dataframes for one flow condition
        # = list containing the original dataframe
        data_all = [df_proj.copy()]

    return data_all, shear_list


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


def fill_missing_timepoints(
    data_crop: pd.DataFrame,
    full_timepoint_range: tuple[float, float],
) -> pd.DataFrame:
    """
    Fill missing timepoints in dataframe for a single crop using NaN padding.
    Note: this function resets the index of the input crop-based dataframe.

    Parameters
    ----------
    data_crop
        DataFrame for a single crop.
    full_timepoint_range
        Tuple specifying the full range of timepoints (start, end) for the dataset.

    Returns
    -------
    :
        DataFrame with missing timepoints filled with NaNs.
    """

    # use full timepoint range for the dataset to ensure that all timepoints are
    # included
    all_timepoints = np.arange(full_timepoint_range[0], full_timepoint_range[1] + 1)

    # reindex dataframe to include all timepoints in full range
    data_crop_filled = data_crop.set_index(Column.TIMEPOINT).reindex(all_timepoints)

    # reset index to restore timepoint column
    data_crop_filled = data_crop_filled.reset_index()

    return data_crop_filled
