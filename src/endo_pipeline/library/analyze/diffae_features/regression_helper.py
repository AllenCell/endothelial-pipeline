from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.endo_pipeline.configs import DatasetConfig
from src.endo_pipeline.library.analyze.kramersmoyal import get_km_kernel


def get_traj_by_flow(
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


def masked_vector_field(f: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For the vector field f over grid x, mask out
    f at points x where f(x) is NaN.

    Inputs:
    - f: numpy array (n_1 x n_2 x ... x n_ndim x ndim),
        ndim-D vector field evaluated on a meshgrid
    - x: numpy array (n_1 x n_2 x ... x n_ndim x ndim),
        ndim-D meshgrid where vector field is evaluated

    Outputs:
    - f_mask: numpy array (m x ndim), masked vector field
        flattened to 2D array (m = number of non-NaN points)
    - x_mask: numpy array (m x ndim), masked meshgrid
        flattened to 2D array (m = number of non-NaN points)
    """
    # mask out NaN values in F
    mask = np.where(np.isfinite(f))
    ndim = f.shape[-1]

    # mask and flatten F and X over grid
    x_mask = x[mask].reshape((-1, ndim))
    f_mask = f[mask].reshape((-1, ndim))

    return f_mask, x_mask


def train_test_all(
    x: list[np.ndarray],
    drift: list[np.ndarray],
    diffusion: list[np.ndarray],
    train_frac: float = 0.8,
    seed: int = 47,
) -> tuple:
    """
    Split feature data from a given dataset into training
    and testing sets for each flow condition present in the dataset.

    Inputs:
    - x: list of numpy arrays, each array contains
        the points in feature space
        for a single flow condition
    - drift: list of numpy arrays, each array contains
        the drift estimates for each point in x
        for a single flow condition
    - diffusion: list of numpy arrays, each array contains
        the diffusion estimates for each point in x
        for a single flow condition
    - train_frac: fraction of data to use for training
        (default = 0.8)
    - seed: random seed for train/test split
        (default = 47)

    Outputs:
    - x_train: points in feature space corresponding to
        the drift and diffusion estimates in the training sets
    - x_test: points in feature space corresponding to
        the drift and diffusion estimates in the test sets
    - y_train: training data for drift estimates
    - y_test: test data for drift estimates
    - v_train: training data for diffusion estimates
    - v_test: test data for diffusion estimates
    """
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    v_train = []
    v_test = []

    num_flow = len(x)

    # get train/test split for each flow condition
    for j in range(num_flow):
        x_train_, x_test_, y_train_, y_test_ = train_test_split(
            x[j], drift[j], train_size=train_frac, random_state=seed + j
        )
        # same random seed to get same x points for train and test
        _, _, v_train_, v_test_ = train_test_split(
            x[j], diffusion[j], train_size=train_frac, random_state=seed + j
        )
        x_train.append(x_train_)
        x_test.append(x_test_)
        y_train.append(y_train_)
        y_test.append(y_test_)
        v_train.append(v_train_)
        v_test.append(v_test_)

    # concatenate all data into one array, one train/test for all flow conditions
    x_train = np.concatenate(x_train)
    x_test = np.concatenate(x_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    v_train = np.concatenate(v_train)
    v_test = np.concatenate(v_test)

    return x_train, x_test, y_train, y_test, v_train, v_test


def get_stationary_hist(
    stationary_data: pd.DataFrame,
    pc_column_names: list[str],
    bins: list,
) -> np.ndarray:
    """
    Get stationary histogram of data.

    Inputs:
    - stationary_data: pandas DataFrame containing the
        dataset of interest restricted to stationary frames
    - pc_column_names: list of strings, names of the
        columns in the DataFrame that contain the
        principal component features (feature columns)
    - bins: list of number of bins in each dimension
        (list of length ndim, where ndim is the
        number of dimensions of the feature space)

    Outputs:
    - p_hist: numpy array, stationary histogram
        of the data in feature space
    """
    ndim = len(pc_column_names)

    # call 1D or 2D histogram function based on number of dimensions
    if ndim == 2:
        # data frame_number > frame_index, all rows, select pcs
        p_hist, _, _ = np.histogram2d(
            stationary_data[pc_column_names[0]],
            stationary_data[pc_column_names[1]],
            bins,
            density=True,
        )
    elif ndim == 1:
        p_hist, _ = np.histogram(
            stationary_data[pc_column_names[0]],
            bins[0],
            density=True,
        )
    else:
        raise ValueError("Only 1D or 2D data currently supported.")

    return p_hist


def save_train_test(train_test_dict: dict, savedir: Path) -> None:
    """
    Save train test data to file in savedir, using `numpy.savez` function.

    Inputs:
    - train_test_dict: dict, dictionary containing train and test data (numpy arrays).
    - savedir: Path, directory to save the file.

    Outputs:
    - None, save the file to savedir.
    """
    np.savez(savedir / "train_test_data", **train_test_dict)


def load_train_test(file_path: Path) -> dict:
    """
    Load train test data from file_path.

    Inputs:
    - file_path: Path, path to the file containing train test data.

    Outputs:
    - train_test_dict: dict, dictionary containing train and test data (numpy arrays).
    """
    return dict(np.load(file_path, allow_pickle=True))
