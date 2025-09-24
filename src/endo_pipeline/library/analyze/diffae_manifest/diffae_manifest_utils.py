import logging

import numpy as np
import pandas as pd

from endo_pipeline.configs import DatasetConfig, get_frame_after_flow_change, load_dataset_config

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
        e.g., "48hr_High_Shear_Stress_30_dyncm2".

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

    # initialize dictionary to store descriptions
    description_dict = {}
    for dataset_name in list_of_datasets:
        dataset_config = load_dataset_config(dataset_name)  # get dataset info from data_config.yaml
        flow_conditions = dataset_config.flow_conditions  # get flow conditions for dataset
        num_flows = len(flow_conditions)  # number of flow conditions in dataset

        # get shear stress for each flow condition,
        # last element in each list in flow_config
        shear_stress = [int(flow_conditions[i].shear_stress) for i in range(num_flows)]
        shear_stress_strings: list[str] = []
        if simple:  # if simple description, use qualitative description of shear stress level
            shear_stress_strings = []
            for i, shear in enumerate(shear_stress):
                shear_stress_str = dataset_config.shear_stress_regime

                # have to parse differently if multiple flow conditions
                if num_flows > 1:
                    if i == 0:
                        shear_stress_str = shear_stress_str.split("_")[0]
                    else:
                        shear_stress_str = shear_stress_str.split("_")[-1]

                if include_duration:
                    duration_in_frames = flow_conditions[i].stop - flow_conditions[i].start
                    duration_in_hours = int(duration_in_frames * 5 / 60)
                    shear_stress_str = f"{duration_in_hours}hr_{shear_stress_str}"

                if include_shear_stress:
                    shear_stress_str = f"{shear_stress_str}_{int(shear)}dyncm2"

                shear_stress_strings.append(shear_stress_str)
        else:
            for i, shear in enumerate(shear_stress):
                shear_stress_str = f"{int(shear)}_dyncm2"
                if include_duration:
                    duration_in_frames = flow_conditions[i].stop - flow_conditions[i].start
                    duration_in_hours = int(duration_in_frames * 5 / 60)  # convert to hours
                    shear_stress_str = f"{duration_in_hours}hr_{shear_stress_str}"
                shear_stress_strings.append(shear_stress_str)

        description = "_".join(
            [shear_stress_strings[i] for i in range(num_flows)]
        )  # concatenate time and shear rate for each flow condition
        description_dict[dataset_name] = description  # add description to dictionary

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


def get_timepoints_for_plotting_pcs(
    list_of_datasets: list[DatasetConfig],
    restrict_no_flow: bool = True,
    no_flow_name: str = "20241217_20X",
) -> dict:
    """
    Get timepoints for plotting scatter plot in PC space of data used to fit PCA.

    Used to remove later block of timepoints from the 20241217_20X no flow dataset for
    generating "simplified" scatter plots for the 2025 SAC presentation.
    """
    # initialize dictionary to store timepoints for each dataset
    timepoints_to_use = {}

    for dataset_config in list_of_datasets:
        # get range of valid timepoints for each dataset
        # loaded from dataset config
        valid_timepoints = dataset_config.valid_timepoints

        # if no valid timepoints are specified, use all timepoints
        if valid_timepoints is None:
            timepoints_list = [[0, dataset_config.flow_conditions[-1].stop]]

        # otherwise, get the start and stop timepoints
        else:
            starts = valid_timepoints.start
            stops = valid_timepoints.stop
            timepoints_list = []
            for start, stop in zip(starts, stops, strict=True):
                # hard coded because this is the no-flow dataset that
                # we are using for fitting the PCs, and specifically
                # the one with the two sets of timepoints
                # if this changes, we can updated this to not be
                # hardcoded (i.e., check if shear stress is 0 in config)
                if dataset_config.name == no_flow_name and restrict_no_flow:
                    # restrict to only first set of no flow timepoints
                    if start == 0:
                        timepoints_list.append([start, stop])
                    else:
                        continue
                else:
                    timepoints_list.append([start, stop])
        timepoints_to_use[dataset_config.name] = timepoints_list
    return timepoints_to_use


def get_valid_subset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Filter dataframe to only include valid timepoints for a given dataset.

    **Input dataframe**

    The input dataframe should have a column "frame_number" indicating the timepoint of each crop.

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
        valid_subset = df.frame_number.isin(tps)
        df["valid"] = valid_subset
    return df[df.valid]


def get_pc_column_names(df: pd.DataFrame, pc_axes: list[int] | None = None) -> list[str]:
    """Get the names of the PC columns in the DataFrame."""

    # get all columns that start with "pc"
    pc_column_names = [c for c in df.columns if c.startswith("pc")]
    pc_column_names = sorted(pc_column_names, key=lambda x: int(x[-1]))

    if pc_axes is not None:
        # get only the specified PC axes
        pc_column_names = [pc_column_names[i] for i in pc_axes]

    return pc_column_names


def get_feature_column_names(df: pd.DataFrame) -> list:
    """Get the names of the latent feature columns in the DataFrame."""
    feature_column_names = [c for c in df.columns if c.startswith("feat_")]
    feature_column_names = sorted(feature_column_names, key=lambda x: int(x.split("_")[1]))
    return feature_column_names
