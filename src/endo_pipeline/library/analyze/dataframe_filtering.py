"""Module for methods that filter dataframes for analysis based on certain criteria."""

import logging

import pandas as pd

from endo_pipeline.configs import (
    DatasetConfig,
    FlowCondition,
    PositionAnnotation,
    TimepointAnnotation,
    get_all_unannotated_timepoints,
    get_frame_after_flow_change,
    get_unannotated_positions,
)
from endo_pipeline.library.analyze.dataframe_validation import (
    check_dataframe_dataset_matches_dataset_config,
    check_required_columns_in_dataframe,
)
from endo_pipeline.settings.column_names import ColumnName as Column

logger = logging.getLogger(__name__)


def filter_dataframe_by_track_length(
    dataframe: pd.DataFrame, minimum_track_length: int
) -> pd.DataFrame:
    """
    Filter dataframe to only include tracks above a minimum track length.

    The expected column name for track length is set as Column.TRACK_LENGTH.
    There is a check to ensure that the specified track length column is present
    in the dataframe, and if not, a ValueError is raised by the method
    `check_required_columns_in_dataframe`.

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
    minimum_track_length
        Minimum track length to filter tracks.

    Returns
    -------
    :
        Filtered DataFrame containing only tracks with length >=
        minimum_track_length.
    """

    logger.debug(
        "Filtering dataframe to only include tracks with length >= [ %s ] timepoints.",
        minimum_track_length,
    )
    logger.debug("Dataframe length before filtering: [ %s ] rows.", len(dataframe))
    # check that required columns are present in dataframe
    check_required_columns_in_dataframe(dataframe, [Column.TRACK_LENGTH])
    dataframe_filtered = dataframe[dataframe[Column.TRACK_LENGTH] >= minimum_track_length]

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

    # check that dataframe is restricted to a single dataset, and that the
    # dataset name in the dataframe matches the dataset name for the provided
    # dataset config
    check_dataframe_dataset_matches_dataset_config(dataframe, dataset_config)

    # get positions and timepoints to include based on annotations
    only_include_positions = get_unannotated_positions(dataset_config, position_annotations)
    only_include_frames = get_all_unannotated_timepoints(dataset_config, timepoint_annotations)

    # filter dataframe to only include non-annotated positions
    dataframe_exclude_positions = dataframe[dataframe[Column.POSITION].isin(only_include_positions)]
    # filter dataframe to only include non-annotated timepoints
    df_filtered_list = []
    for position, df_position in dataframe_exclude_positions.groupby(Column.POSITION):
        include_frames_for_position = only_include_frames.get(position, [])
        df_position_filtered = df_position[
            df_position[Column.TIMEPOINT].isin(include_frames_for_position)
        ]
        df_filtered_list.append(df_position_filtered)
    dataframe_filtered = pd.concat(df_filtered_list, ignore_index=True)

    return dataframe_filtered


def filter_dataframe_to_steady_state(
    dataframe: pd.DataFrame, dataset_config: DatasetConfig
) -> pd.DataFrame:
    """
    Filter dataframe to only include timepoints "steady state" timepoints.

    Filtering is done by removing timepoints that are annotated as
    NOT_STEADY_STATE in the dataset config.

    Parameters
    ----------
    dataframe
        Dataframe of features for one dataset.
    dataset_config
        Dataset config for the dataset.
    """

    # note: don't need to do dataframe validation checks here since those will
    # be done in the method `filter_dataframe_by_annotations` that is called
    # within this method

    # To just filter to steady state timepoints, we can use the more general
    # method `filter_dataframe_by_annotations` with
    # timepoint_annotations=[NOT_STEADY_STATE] and position_annotations=[]
    # (i.e., don't do any additional filtering based on position annotations)
    dataframe_steady_state = filter_dataframe_by_annotations(
        dataframe=dataframe,
        dataset_config=dataset_config,
        position_annotations=[],
        timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
    )
    return dataframe_steady_state


def filter_dataframe_by_flow_condition(
    dataframe: pd.DataFrame, dataset_config: DatasetConfig, flow_condition: FlowCondition
) -> pd.DataFrame:
    """
    Filter dataframe to only include timepoints corresponding to a specified flow condition.

    Parameters
    ----------
    dataframe
        Dataframe of features for one dataset.
    dataset_config
        Dataset config for the dataset.
    flow_condition
        Flow condition to filter by.

    Returns
    -------
    :
        Dataframe filtered to only include timepoints corresponding to the specified flow condition.
    """

    # check that required columns are present
    required_columns = [Column.DATASET, Column.TIMEPOINT]
    check_required_columns_in_dataframe(dataframe, required_columns)

    # check that dataframe is restricted to a single dataset, and that the
    # dataset name in the dataframe matches the dataset name for the provided
    # dataset config
    check_dataframe_dataset_matches_dataset_config(dataframe, dataset_config)

    # get flow condition information from dataset config
    flow_conditions_in_dataset = dataset_config.flow_conditions

    # Check that provided flow condition actually in the dataset config. If not, raise an error.
    if flow_condition not in flow_conditions_in_dataset:
        raise ValueError(
            f"Specified flow condition [ {flow_condition} ] does not match any of the flow conditions"
            f" in the dataset config [ {flow_conditions_in_dataset} ] for dataset {dataset_config.name}."
        )

    if len(flow_conditions_in_dataset) == 1:
        # If only one flow condition in dataset, return the original dataframe
        # since all timepoints correspond to the specified flow condition (i.e.,
        # the only flow condition in the dataset).
        return dataframe.copy()
    else:
        # multi-flow condition dataset: need to filter to timepoints
        # corresponding to specified flow condition
        change_frame = get_frame_after_flow_change(dataset_config)
        if flow_condition == flow_conditions_in_dataset[0]:
            # if first flow condition specified, filter to timepoints before
            # the flow change frame
            return dataframe[dataframe[Column.TIMEPOINT] < change_frame].copy()
        else:
            # if second flow condition specified, filter to timepoints after
            # the flow change frame
            return dataframe[dataframe[Column.TIMEPOINT] >= change_frame].copy()


def split_dataframe_by_flow(
    dataframe: pd.DataFrame, dataset_config: DatasetConfig
) -> tuple[list[pd.DataFrame], list[float]]:
    """
    Parse a dataframe of features for one dataset into separate dataframes for
    each flow condition based on the dataset config.

    If there is only one flow condition, this method returns a lists of length 1
    containing the original dataframe and single shear stress value.

    The dataframe should have columns for:
        - Column.DATASET: dataset name for each row of data, which should be the
          same across all rows of the dataframe.
        - Column.TIMEPOINT: timepoint/frame number for each row of data.

    Parameters
    ----------
    dataframe
        DataFrame containing feature data for one dataset.
    dataset_config
        DatasetConfig object for the given dataset.

    Returns
    -------
    :
        List of DataFrames, each containing the feature data for one flow
        condition.
    :
        List of shear stress values for each flow condition.
    """
    # check that required columns are present
    required_columns = [Column.DATASET, Column.TIMEPOINT]
    check_required_columns_in_dataframe(dataframe, required_columns)

    # check that dataframe is restricted to a single dataset, and that the
    # dataset name in the dataframe matches the dataset name for the provided
    # dataset config
    check_dataframe_dataset_matches_dataset_config(dataframe, dataset_config)

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
        # separate data into two dataframes based on
        # frame number where flow condition changes
        data_flow1 = dataframe[dataframe[Column.TIMEPOINT] < change_frame].copy()
        data_flow2 = dataframe[dataframe[Column.TIMEPOINT] >= change_frame].copy()
        # return list of dataframes for each flow condition
        data_all = [data_flow1, data_flow2]
    # else, there is only one flow condition
    else:
        # list of dataframes for one flow condition
        # = list containing the original dataframe
        data_all = [dataframe.copy()]

    return data_all, shear_list
