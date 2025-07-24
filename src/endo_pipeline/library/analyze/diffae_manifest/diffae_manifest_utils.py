import numpy as np
import pandas as pd

from src.endo_pipeline.configs import DatasetConfig, load_dataset_config


def get_dataset_descriptions(list_of_datasets: list[str], simple: bool = False) -> dict:
    """
    Get descriptive metadata for each dataset given in the list of datasets.

    Describes the experimental conditions for each dataset,
        e.g., "48_hours_at_30_dyncm2".

    Inputs:
    - list_of_datasets: list, list of dataset names to get descriptions for
        - Each string should match the appropriate dataset name in data_config.yaml
    - simple (optional): bool, whether to use simple description (e.g., "48hr_High")


    Outputs:
    - description_dic: dict, dictionary of dataset names and their descriptive metadata
    """

    # initialize dictionary to store descriptions
    description_dic = {}
    for dataset_name in list_of_datasets:
        data_config = load_dataset_config(dataset_name)  # get dataset info from data_config.yaml
        flow_config = data_config.flow  # get flow conditions for dataset
        num_flows = len(flow_config)  # number of flow conditions in dataset

        # get shear rate for each flow condition,
        # last element in each list in flow_config
        shear_rate = [int(flow_config[i][-1]) for i in range(num_flows)]
        if simple:  # if simple description, use qualitative description of shear stress level
            shear_rate_str = []
            for shear in shear_rate:
                if shear >= 20:
                    shear_rate_str.append("High")
                elif shear > 7:
                    shear_rate_str.append(f"Intermediate_{int(shear)}")
                elif shear > 0:
                    shear_rate_str.append("Low")
                else:
                    shear_rate_str.append("No")
        else:
            shear_rate_str = [
                f"{int(i)}_dyncm2" for i in shear_rate
            ]  # convert shear rates to strings

        time_str = [
            f"{int((flow_config[i][1]-flow_config[i][0])*5/60)}hr" for i in range(num_flows)
        ]  # get duration of each flow condition in hours
        description = "_".join(
            [time_str[i] + "_" + shear_rate_str[i] for i in range(num_flows)]
        )  # concatenate time and shear rate for each flow condition
        description_dic[dataset_name] = description  # add description to dictionary

    return description_dic


def split_dataset_by_flow(
    df_proj: pd.DataFrame, dataset_config: DatasetConfig, verbose: bool = True
) -> tuple[list, list]:
    """
    Get crop-based feature data (Diffusion AE output) for
    different flow conditions present in a dataset.

    Inputs:
    - df_proj: pandas dataframe containing the dataset of interest,
        projected onto all principal component axes
        (change of basis, no dimensionality reduction)
    - dataset_config: DatasetConfig object containing dataset configuration
        (used to get flow information)
    - verbose: boolean, if True, print information about flow conditions

    Outputs:
    - data_all: list of dataframes, each containing
        the feature data for one flow condition
    - shear_list: list of shear stress conditions for each flow condition

    If there is only one flow condition, data_all and shear_list
    are still lists (of length 1), respectively containing the
    original dataframe and single shear stress condition.
    """

    # load flow information from data_config.yaml
    flow_info = dataset_config.flow

    # split out data by flow condition,
    # starting with first flow condition
    first_shear = float(flow_info[0][-1])
    # initialize list of shear stress conditions
    shear_list = [first_shear]
    # if there is a change in flow condition
    if len(flow_info) > 1:
        # get frame number where flow condition
        # changes (reported in hours in data_config.yaml)
        change_frame = flow_info[0][-1]
        # get second shear stress condition
        second_shear = float(flow_info[1][-1])
        shear_list.append(second_shear)
        if verbose:
            print(f"Shear stress {first_shear} dyn/cm^2 until frame {change_frame}")
            print(f"Shear stress {second_shear} dyn/cm^2 after frame {change_frame} \n")
        # separate data into two dataframes based on
        # frame number where flow condition changes
        data_flow1 = df_proj[df_proj["frame_number"] < change_frame].copy()
        data_flow2 = df_proj[df_proj["frame_number"] >= change_frame].copy()
        # return list of dataframes for each flow condition
        data_all = [data_flow1, data_flow2]
    # else, there is only one flow condition
    else:
        if verbose:
            print("Constant shear stress at", first_shear, "dyn/cm^2 \n")
        # list of dataframes for one flow condition
        # = list containing the original dataframe
        data_all = [df_proj.copy()]

    return data_all, shear_list


def get_traj_and_diff(data: pd.DataFrame, pc_column_names: list) -> tuple[list, list]:
    """
    Get list of per-crop trajectories, the corresponding
    displacement vectors, and time differences along
    the trajectory for each crop in the dataset.

    Inputs:
    - data: pandas DataFrame with columns for each feature.
        Should have a column for time and a column for
        the crop index. This data should be for one
        dataset and one flow condition.
    - pc_column_names: list of strings, names of the
        columns in the DataFrame that contain the
        PC features (feature columns).

    Outputs:
    - traj_list: list of numpy arrays, each array is the
        trajectory of a single crop in feature space
    - d_traj_list: list of numpy arrays, each array
        is the displacement vectors along that trajectory
        for a single crop in feature space
    """
    if "frame_number" not in data.columns:
        raise ValueError("Data must have a column for time")
    if "crop_index" not in data.columns:
        raise ValueError("Data must have a column for crop_index")

    # get list of unique crop indices
    crop_list = data["crop_index"].unique()

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
    Get timepoints for plotting scatter plot in PC
    space of data used to fit PCA.

    Used to remove later block of timepoints from the
    20241217_20X no flow dataset for generating "simplified"
    scatter plots for the 2025 SAC presentation.
    """
    # initialize dictionary to store timepoints for each dataset
    timepoints_to_use = {}

    for dataset_config in list_of_datasets:
        # get range of valid timepoints for each dataset
        # loaded from dataset config
        timepoint_dict = dataset_config.valid_timepoints

        # if no valid timepoints are specified, use all timepoints
        if timepoint_dict is None:
            timepoints_list = [[0, dataset_config.flow[0][1]]]

        # otherwise, get the start and stop timepoints
        else:
            starts = timepoint_dict.start
            stops = timepoint_dict.stop
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


def get_valid_subset(df: pd.DataFrame, dataset_name: str, verbose: bool = True) -> pd.DataFrame:
    """
    Select timepoints from a dataframe annotated as valid
    if annotation is present, otherwise use all timepoints.

    Inputs:
    - df: pd.DataFrame, containing the metadata for the dataset name and timepoints
    - dataset_name: str, name of the dataset to get valid timepoints for

    Outputs:
    - df: pd.DataFrame, subset of the input dataframe containing only the valid timepoints
    """
    df["valid"] = False
    # check that the necessary datasets are present for fitting PCA
    valid_timepoints = load_dataset_config(dataset_name).valid_timepoints
    if valid_timepoints is None:
        if verbose:
            print(f"Using all timepoints from dataset {dataset_name} for analysis")
        df["valid"] = True
    else:
        if verbose:
            print(f"Valid timepoints for dataset {dataset_name}: ")
        tps = []
        for start, stop in zip(valid_timepoints.start, valid_timepoints.stop, strict=True):
            tps.extend(list(range(start, stop + 1)))
            if verbose:
                print(f"   - {start} to {stop}")
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
    """
    Extract columns corresponding to DiffAE model
    features from dataframe (loaded DiffAE manifest).
    """
    feature_column_names = [c for c in df.columns if c.startswith("feat_")]
    feature_column_names = sorted(feature_column_names, key=lambda x: int(x.split("_")[1]))
    return feature_column_names
