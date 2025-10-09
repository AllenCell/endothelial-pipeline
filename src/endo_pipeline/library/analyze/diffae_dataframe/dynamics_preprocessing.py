import logging

import numpy as np
import pandas as pd

from endo_pipeline.configs import DatasetConfig, get_frame_after_flow_change

logger = logging.getLogger(__name__)


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
        data_flow1 = df_proj[df_proj["frame_number"] < change_frame].copy()
        data_flow2 = df_proj[df_proj["frame_number"] >= change_frame].copy()
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
    if "frame_number" not in data.columns:
        logger.error("Data must have the column [ frame_number ] to indicate timepoints.")
        raise ValueError("Data must have the column [ frame_number ] to indicate timepoints.")
    if "crop_index" not in data.columns:
        logger.error("Data must have the column [ crop_index ] to indicate unique crops.")
        raise ValueError("Data must have the column [ crop_index ] to indicate unique crops.")

    # get list of unique crop indices
    crop_list = data["crop_index"].unique().tolist()

    # initialize lists for storing outputs
    traj_list = []
    d_traj_list = []

    # loop over each crop in the dataset
    for crop in crop_list:
        # get data for each crop, sorted by time
        data_crop = data[data["crop_index"] == crop].sort_values(by="frame_number")

        # get displacement vectors and time differences for each crop
        d_traj = np.diff(data_crop[pc_column_names].values, axis=0)

        # append data to lists:
        # trajectory and displacement vectors
        # leave off last timepoint for trajectory
        traj_list.append(data_crop[pc_column_names].values)
        d_traj_list.append(d_traj)

    return traj_list, d_traj_list
