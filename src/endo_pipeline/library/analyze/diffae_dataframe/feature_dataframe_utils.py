import logging

import numpy as np
import pandas as pd
from deprecated import deprecated  # type:ignore[import-untyped]

from endo_pipeline.configs import DatasetConfig, get_frame_after_flow_change, load_dataset_config
from endo_pipeline.settings import ColumnName

logger = logging.getLogger(__name__)


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


def split_dataset_by_flow(
    df_proj: pd.DataFrame, dataset_config: DatasetConfig
) -> tuple[list, list]:
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
        data_flow1 = df_proj[df_proj[ColumnName.TIMEPOINT] < change_frame].copy()
        data_flow2 = df_proj[df_proj[ColumnName.TIMEPOINT] >= change_frame].copy()
        # return list of dataframes for each flow condition
        data_all = [data_flow1, data_flow2]
    # else, there is only one flow condition
    else:
        logger.debug("Constant shear stress [ %s ] dyn/cm^2", first_shear)
        # list of dataframes for one flow condition
        # = list containing the original dataframe
        data_all = [df_proj.copy()]

    return data_all, shear_list


def get_traj_and_diff(data: pd.DataFrame, pc_column_names: list) -> tuple[list, list]:
    """
    Get trajectories and displacement vectors for each crop in feature space.

    **Input dataframe**

    The input dataframe should have columns for:
    - frame_number: timepoint of the crop
    - crop_index: unique index for each crop
    - columns for each feature (e.g., pc0, pc1, pc2, ...) matching input ``pc_column_names``

    Parameters
    ----------
    data
        DataFrame with columns for each feature.
    pc_column_names
        List of column names corresponding to the PC features in the DataFrame.

    Returns
    -------
    :
        List of individual crop trajectories in feature space.
    :
        List of displacement vectors along each trajectory in feature space.
    """
    if ColumnName.TIMEPOINT not in data.columns:
        logger.error(
            "Data must have the column [ %s ] to indicate timepoints.", ColumnName.TIMEPOINT
        )
        raise ValueError(
            f"Data must have the column [ {ColumnName.TIMEPOINT} ] to indicate timepoints."
        )
    if ColumnName.CROP_INDEX not in data.columns:
        logger.error(
            "Data must have the column [ %s ] to indicate unique crops.", ColumnName.CROP_INDEX
        )
        raise ValueError(
            f"Data must have the column [ {ColumnName.CROP_INDEX} ] to indicate unique crops."
        )

    # get list of unique crop indices
    crop_list = data[ColumnName.CROP_INDEX].unique().tolist()

    # initialize lists for storing outputs
    traj_list = []
    d_traj_list = []

    # loop over each crop in the dataset
    for crop in crop_list:
        # get data for each crop, sorted by time
        data_crop = data[data[ColumnName.CROP_INDEX] == crop].sort_values(by=ColumnName.TIMEPOINT)

        # get displacement vectors and time differences for each crop
        d_traj = np.diff(data_crop[pc_column_names].values, axis=0)

        # append data to lists:
        # trajectory and displacement vectors
        # leave off last timepoint for trajectory
        traj_list.append(data_crop[pc_column_names].values)
        d_traj_list.append(d_traj)

    return traj_list, d_traj_list


@deprecated(
    """
This method is deprecated and will be removed. The recommended alternative is:

    from endo_pipeline.library.analyze.diffae_dataframe import (
        remove_annotated_timepoints_and_positions,
    )

    df_valid = remove_annotated_timepoints_and_positions(
        df, exclude_cell_piling=True, exclude_not_steady_state=True
    )
"""
)
def get_valid_subset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Filter dataframe to only include valid timepoints for a given dataset.

    Parameters
    ----------
    df
        DataFrame of feature data for a given dataset.
    dataset_name
        Name of the given dataset.
    """
    df["valid"] = False
    # check that the necessary datasets are present for fitting PCA
    valid_timepoints = load_dataset_config(dataset_name).valid_timepoints
    if valid_timepoints is None:
        logger.debug("Using all timepoints from dataset [ %s ] for analysis", dataset_name)
        df["valid"] = True
    else:
        tps = []
        for start, stop in zip(valid_timepoints.start, valid_timepoints.stop, strict=True):
            tps.extend(list(range(start, stop + 1)))
            logger.debug(
                "Using timepoints [ %s - %s ] from dataset [ %s ] for analysis",
                start,
                stop,
                dataset_name,
            )
        valid_subset = df[ColumnName.TIMEPOINT].isin(tps)
        df["valid"] = valid_subset
    return df[df.valid]
