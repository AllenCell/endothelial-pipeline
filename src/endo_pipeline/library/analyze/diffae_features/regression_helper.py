from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe import (
    get_dataframe_for_dynamics_workflows,
    get_traj_and_diff,
    split_dataset_by_flow,
)
from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from endo_pipeline.library.analyze.numerics import get_bins
from endo_pipeline.library.visualize.diffae_features import feature_viz
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES, ColumnName


def _kramers_moyal_train_test_one_dataset(
    df_proj: pd.DataFrame,
    dataset_name: str,
    pc_axes: list,
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
    Generate train/test sets for regression on Kramers-Moyal estimates from one dataset.

    The first Kramers-Moyal coefficient corresponds to the drift, and the second to the diffusion.

    This function is called by ``_build_kramers_moyal_train_test`` in a loop over all datasets
    being processed by the ``build_sde_train_test`` workflow.

    **Method output**

    This function returns the following outputs as a tuple:

    - x_train: training set of pc coordinates for Kramers-Moyal estimates
    - x_test: test set of pc coordinates for Kramers-Moyal estimates
    - y_train: training set of drift estimates
    - y_test: test set of drift estimates
    - v_train: training set of diffusion estimates
    - v_test: test set of diffusion estimates
    - u_train: training set of flow conditions (shear stresses)
    - u_test: test set of flow conditions (shear stresses)

    Parameters
    ----------
    df_proj
        Dataframe containing the PC features for a single dataset.
    dataset_name
        Name of the dataset (used to split out data by flow condition via dataset config).
    pc_axes
        List of principal component axes to use for analysis.
    num_bins
        List of number of bins to use for histogramming data.
    dt
        Time step between data points.
    train_frac
        Fraction of data to use for training in train/test split.
    fig_savedir
        Directory to save figures.
    kernel_params
        Dictionary of parameters for kernel regression method.
    """

    # for extracting just the axes (specified via PCs)
    # we want from the resulting dataframe
    # e.g., if we are just analyzing the first two PCs,
    # we want to extract columns 'pc_1' and 'pc_2'
    ndim = len(pc_axes)
    pc_column_names = [DIFFAE_PC_COLUMN_NAMES[pc] for pc in pc_axes]

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
            frame_max = df_by_flow[j][ColumnName.TIMEPOINT].max()
            frame_cutoff = frame_max - 100
            stationary_data = df_by_flow[j][df_by_flow[j][ColumnName.TIMEPOINT] > frame_cutoff]
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
        fig = feature_viz.plot_km(centers, kmc, pc_axes, shear_list[j])[0]
        save_plot_to_path(
            fig,
            fig_savedir,
            f"kmcs_all_{dataset_name}_flow_{j}",
        )

        # quiver and streamplot of drift vector field
        if ndim == 2:
            fig = feature_viz.plot_km_drift_2d(centers, kmc, pc_axes, shear_list[j])[0]
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
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    pc_axes: list[int],
    num_bins: list[int],
    dt: float,
    fig_savedir: Path,
    train_frac: float = 0.8,
    kernel_params: dict | None = None,
) -> dict:
    """
    Generate train/test sets for regression on Kramers-Moyal estimates from multiple datasets.

    The first Kramers-Moyal coefficient corresponds to the drift, and the second to the diffusion.

    This function is called by the ``main`` function in the workflow ``build_sde_train_test``
    to generate the train/test sets for the regression model fitting and evaluation
    of the dynamical systems model for the Diff AE features.

    **Method output**

    This function returns a dictionary with the train/test split data as numpy arrays.
    It contains the following keys/items:
    - "x_train": training set of pc coordinates for Kramers-Moyal estimates
    - "x_test": test set of pc coordinates for Kramers-Moyal estimates
    - "y_train": training set of drift estimates
    - "y_test": test set of drift estimates
    - "v_train": training set of diffusion estimates
    - "v_test": test set of diffusion estimates
    - "u_train": training set of flow conditions (shear stresses)
    - "u_test": test set of flow conditions (shear stresses)

    These arrays are concatenated across all datasets specified in `dataset_names`.

    Parameters
    ----------
    dataset_names
        List of dataset names to process.
    dataframe_manifest
        Dataframe manifest containing locations of the feature dataframes.
    pca
        PCA object for projecting feature data onto principal component axes.
    pc_axes
        List of principal component axes to use for analysis.
    num_bins
        List of number of bins to use for histogramming data.
    dt
        Time step between data points.
    fig_savedir
        Directory to save figures.
    train_frac
        Fraction of data to use for training in train/test split.
    kernel_params
        Dictionary of parameters for kernel regression method.
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
        df_proj = get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest, pca=pca)

        # get train test split for this dataset
        x_train, x_test, y_train, y_test, v_train, v_test, u_train, u_test = (
            _kramers_moyal_train_test_one_dataset(
                df_proj,
                dataset_name,
                pc_axes,
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
    Mask out values of a vector field that are NaN.

    **Method inputs**

    The input vector field is assumed to be evaluated on a meshgrid, with the corresponding
    array having shape (n_1, n_2, ..., n_ndim, ndim), where ndim is the number of dimensions
    of the vector field, and n_i is the number of points along dimension i.

    The array x is the corresponding meshgrid, with the same shape as f. That is, f is the
    array obtained from evaluating the vector field on the meshgrid x.

    **Method outputs**

    The output arrays are flattened to shape (m, ndim), where m is the number of non-NaN points
    in the input vector field f.

    Parameters
    ----------
    f
        Array containing the vector field values on a meshgrid.
    x
        Array containing the meshgrid points corresponding to f.

    Returns
    -------
    :
        The masked vector field values.
    :
        The corresponding meshgrid points.
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
    Create a train/test split of feature data for each flow condition in a dataset.

    **Method output**

    This function returns the following outputs as a tuple:

    - x_train: Concatenated training set of points in feature space
    - x_test: Concatenated test set of points in feature space
    - y_train: Concatenated training set of drift estimates
    - y_test: Concatenated test set of drift estimates
    - v_train: Concatenated training set of diffusion estimates
    - v_test: Concatenated test set of diffusion estimates

    where concatenation is done across all flow conditions.

    Parameters
    ----------
    x
        List of data points corresponding to the drift/diffusion estimates for each flow condition.
    drift
        List of drift estimates for each flow condition.
    diffusion
        List of diffusion estimates for each flow condition.
    train_frac
        Fraction of data to use for training in train/test split.
    seed
        Random seed for train/test split.
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
    Save dictionary of train/test split arrays to specified directory.

    This method uses the numpy ``savez`` function to save the dictionary of
    arrays as a single .npz file.

    Parameters
    ----------
    train_test_dict
        Dictionary containing train and test data (numpy arrays).
    savedir
        Directory to save the file.
    """
    np.savez(savedir / "train_test_data", **train_test_dict)


def load_train_test(file_path: Path) -> dict:
    """
    Load dictionary of train/test split arrays from specified file.

    This will typically be the .npz file saved by the ``save_train_test`` function.

    Parameters
    ----------
    file_path
        Path to the .npz file containing the train/test data.

    Returns
    -------
    :
        Dictionary containing train and test data (numpy arrays).
    """
    return dict(np.load(file_path, allow_pickle=True))
