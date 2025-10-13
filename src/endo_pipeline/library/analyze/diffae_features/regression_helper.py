from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe import (
    get_dataframe_for_dynamics_workflows,
    get_pc_column_names,
    get_traj_and_diff,
    split_dataset_by_flow,
)
from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from endo_pipeline.library.analyze.numerics import get_bins
from endo_pipeline.library.visualize.diffae_features import feature_viz
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings import TIMEPOINT_COLUMN_NAME


def _kramers_moyal_train_test_one_dataset(
    df_proj: pd.DataFrame,
    dataset_name: str,
    pcs: list,
    num_bins: list,
    dt: float,
    train_frac: float,
    fig_savedir: Path,
    kernel_params: dict | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate train test sets for Kramers-Moyal coefficients
    (drift and diffusion estimates) for one dataset.
    This function is called by _build_kramers_moyal_train_test
    in a loop over all datasets in the dataframe.

    Inputs:
    - df_proj: pandas dataframe containing the dataset of interest,
        projected onto all principal component axes
        (change of basis, no dimensionality reduction)
    - dataset_name: name of the dataset (used to split out data by
        flow condition, acessed via data_config.yaml)
    - pcs: list of principal component axes to project data onto for
        Kramers-Moyal analysis (e.g., [0,1] for first two principal components)
    - num_bins: list of number of bins to use for histogramming
        data to compute the Kramers-Moyal coefficients
        (conditional averages computed in each bin)
    - dt: time step between data points
        (used to compute Kramers-Moyal coefficients)
    - train_frac: fraction of data to use for training
    - fig_savedir: directory to save figures
    - kernel_params: dictionary of parameters for kernel method

    Outputs:
    - x_train: training data for Kramers-Moyal coefficients
        (drift and diffusion estimates) from the given dataset
    - x_test: test data for Kramers-Moyal coefficients from the given dataset
    - y_train: training data for drift estimates from the given dataset
    - y_test: test data for drift estimates from the given dataset
    - v_train: training data for diffusion estimates from the given dataset
    - v_test: test data for diffusion estimates from the given dataset
    - u_train: training flow conditions (shear rates) from the given dataset
    - u_test: test flow conditions from the given dataset
    """

    # for extracting just the axes (specified via PCs)
    # we want from the resulting dataframe
    # e.g., if we are just analyzing the first two PCs,
    # we want to extract columns 'pc1' and 'pc2'
    pc_column_names = get_pc_column_names(df_proj, pcs)
    ndim = len(pcs)

    # split out data by flow condition
    df_by_flow, shear_list = split_dataset_by_flow(df_proj, load_dataset_config(dataset_name))
    num_flow = len(shear_list)

    drift_km = []
    diff_km = []
    x_pts = []

    for j in range(num_flow):
        # if multiple flow conditions,
        # we want to restrict data to
        # only the last 100 frames of
        # that flow condition as our
        # cutoff for stationary data
        if num_flow > 1:
            frame_max = df_by_flow[j][TIMEPOINT_COLUMN_NAME].max()
            frame_cutoff = frame_max - 100
            stationary_data = df_by_flow[j][df_by_flow[j][TIMEPOINT_COLUMN_NAME] > frame_cutoff]
        # else, it is just the whole dataset
        else:
            stationary_data = df_by_flow[j]

        # get list of per-crop trajectories, the corresponding
        # displacement vectors, and time differences
        traj_list, d_traj_list = get_traj_and_diff(stationary_data, pc_column_names=pc_column_names)

        # get bins for histogramming
        # (for drift and diffusion estimates)
        bins, centers = get_bins(num_bins, data=traj_list)

        # get drift and diffusion estimates
        # (Kramers-Moyal coefficients)
        drift_km_, diff_km_ = get_kramers_moyal(
            traj_list,
            d_traj_list,
            bins,
            dt,
            kernel_params=kernel_params,
        )

        # plot drift and diffusion estimates
        kmc = np.concatenate([drift_km_, diff_km_], axis=-1).T
        fig = feature_viz.plot_km(centers, kmc, pcs, shear_list[j])[0]
        save_plot_to_path(
            fig,
            fig_savedir,
            f"kmcs_all_{dataset_name}_flow_{j}",
        )

        # quiver and streamplot of drift vector field
        if ndim == 2:
            fig = feature_viz.plot_km_drift_2d(centers, kmc, pcs, shear_list[j])[0]
            save_plot_to_path(
                fig,
                fig_savedir,
                f"kmcs_drift_{dataset_name}_flow_{j}",
            )

        # remove NaNs from drift and diffusion estimates
        # (bins with no data), get corresponding bin centers as well
        (
            drift_km_masked,
            x_pts_,
        ) = masked_vector_field(drift_km_, np.array(np.meshgrid(*centers)).T)
        diff_km_masked, _ = masked_vector_field(diff_km_, np.array(np.meshgrid(*centers)).T)
        drift_km.append(drift_km_masked)
        diff_km.append(diff_km_masked)
        x_pts.append(x_pts_)

    del df_by_flow  # free up memory

    # get train test split of Kramers-Moyal
    # estimates for each flow condition
    x_train, x_test, y_train, y_test, v_train, v_test = train_test_all(
        x_pts, drift_km, diff_km, train_frac, seed=47
    )

    # get number of training and test points for each flow condition
    num_tot = [x_pts[j].shape[0] for j in range(num_flow)]
    num_train = [int(train_frac * num_tot[j]) for j in range(num_flow)]
    num_test = [num_tot[j] - num_train[j] for j in range(num_flow)]

    # get corresponding flow condition for each training and test point
    u_train = np.concatenate([shear_list[j] * np.ones((num_train[j], 1)) for j in range(num_flow)])
    u_test = np.concatenate([shear_list[j] * np.ones((num_test[j], 1)) for j in range(num_flow)])

    del x_pts, drift_km, diff_km  # free up memory

    return x_train, x_test, y_train, y_test, v_train, v_test, u_train, u_test


def build_kramers_moyal_train_test(
    dataset_names: list[str],
    manifest: DataframeManifest,
    pca: PCA,
    pcs: list[int],
    num_bins: list[int],
    dt: float,
    fig_savedir: Path,
    train_frac: float = 0.8,
    kernel_params: dict | None = None,
) -> dict:
    """
    Build train test sets for Kramers-Moyal coefficients
    (drift and diffusion estimates) for all datasets in the dataframe df.
    This function is called by the main function
    in the workflow to generate the train test sets
    for the regression model fitting and evaluation
    of the dynamical systems model for the Diff AE features.

    Inputs:
    - dataset_names: list of dataset names to use for Kramers-Moyal analysis
    - manifest: manifest of model feature dataframes
    - pca: PCA object used to project data onto principal component axes
    - pcs: list of principal component axes to use for Kramers-Moyal analysis
    - num_bins: list of number of bins to use for histogramming data to compute
        the Kramers-Moyal coefficients
        (conditional averages computed in each bin)
    - dt: time step between data points
        (used to compute Kramers-Moyal coefficients)
    - fig_savedir: directory to save figures
    - train_frac: fraction of data to use for training (default is 0.8)
    - kernel_params: dictionary of parameters for kernel method
        (default is None, which uses default parameters if method is 'kernel')

    Outputs:
    - out_dict: dictionary containing the following keys:
        - 'X_train': training data for Kramers-Moyal coefficients
            (drift and diffusion estimates)
        - 'X_test': test data for Kramers-Moyal coefficients
        - 'Y_train': training data for drift estimates
        - 'Y_test': test data for drift estimates
        - 'V_train': training data for diffusion estimates
        - 'V_test': test data for diffusion estimates
        - 'u_train': training flow conditions (shear rates)
            - passed in as control variable
        - 'u_test': test flow conditions
    The train test sets are concatenated across all datasets in the dataframe.
    """

    # initialize lists to store train test sets for each dataset
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    v_train_list = []
    v_test_list = []
    u_train_list = []
    u_test_list = []

    # for each dataset, generate train test sets for drift and diffusion estimates
    # (Kramers-Moyal coefficients, Y and V, respectively)
    for dataset_name in dataset_names:
        print("**** Generating train/test sets for dataset", dataset_name, "**** \n")

        # load DiffAE feature data from this one dataset
        # and get features projected onto principal component axes
        # as defined by fit PCA object pca.
        df_proj = get_dataframe_for_dynamics_workflows(dataset_name, manifest, pca=pca)

        # get train test split for this dataset
        x_train, x_test, y_train, y_test, v_train, v_test, u_train, u_test = (
            _kramers_moyal_train_test_one_dataset(
                df_proj,
                dataset_name,
                pcs,
                num_bins,
                dt,
                train_frac,
                fig_savedir,
                kernel_params=kernel_params,
            )
        )

        # add train test for this dataset to list
        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        v_train_list.append(v_train)
        v_test_list.append(v_test)
        u_train_list.append(u_train)
        u_test_list.append(u_test)

        del (
            x_train,
            x_test,
            y_train,
            y_test,
            v_train,
            v_test,
            u_train,
            u_test,
        )  # free up memory

    # concatenate all per-dataset train test sets to get final train test sets
    x_train = np.concatenate(x_train_list)
    x_test = np.concatenate(x_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)
    v_train = np.concatenate(v_train_list)
    v_test = np.concatenate(v_test_list)
    u_train = np.concatenate(u_train_list)
    u_test = np.concatenate(u_test_list)

    # store final train test sets in dictionary
    out_dict = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "v_train": v_train,
        "v_test": v_test,
        "u_train": u_train,
        "u_test": u_test,
    }

    return out_dict


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
