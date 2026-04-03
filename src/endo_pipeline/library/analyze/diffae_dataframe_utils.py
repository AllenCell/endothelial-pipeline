import logging
from typing import cast

import pandas as pd

from endo_pipeline.configs import (
    DatasetConfig,
    PositionAnnotation,
    TimepointAnnotation,
    get_all_unannotated_timepoints,
    get_frame_after_flow_change,
    get_unannotated_positions,
)
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.settings.column_names import ColumnName as Column

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
        logger.warning(
            f"Expected dataframe to contain all positions in dataset '{dataset_config.name}', "
            "but it does not."
        )

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
