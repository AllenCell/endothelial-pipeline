from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.endo_pipeline.configs import ModelManifest, load_dataset_config
from src.endo_pipeline.library.analyze.diffae_manifest import (
    get_manifest_for_dynamics_workflows,
    get_pc_column_names,
)
from src.endo_pipeline.library.analyze.numerics.binning import get_bins
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.diffae_features import manifest_viz

from .regression_helper import (
    get_kramers_moyal,
    get_traj_and_diff,
    get_traj_by_flow,
    masked_vector_field,
    train_test_all,
)


def kramers_moyal_train_test_one_dataset(
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
    This function is called by build_kramers_moyal_train_test
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
    df_by_flow, shear_list = get_traj_by_flow(df_proj, load_dataset_config(dataset_name))
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
            frame_max = df_by_flow[j]["frame_number"].max()
            frame_cutoff = frame_max - 100
            stationary_data = df_by_flow[j][df_by_flow[j]["frame_number"] > frame_cutoff]
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
        fig = manifest_viz.plot_km(centers, kmc, pcs, shear_list[j])[0]
        viz_base.save_plot(
            fig,
            filename=fig_savedir / f"kmcs_all_{dataset_name}_flow_{j}",
            format=".png",
            dpi=500,
        )

        # quiver and streamplot of drift vector field
        if ndim == 2:
            fig = manifest_viz.plot_km_drift_2d(centers, kmc, pcs, shear_list[j])[0]
            viz_base.save_plot(
                fig,
                filename=fig_savedir / f"kmcs_drift_{dataset_name}_flow_{j}",
                format=".png",
                dpi=500,
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
    model_manifest_list: list[ModelManifest],
    pca: Pipeline,
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

    Inputs:
    - model_manifest_list: list of ModelManifest objects used
        to load manifest feature data to use for Kramers-Moyal analysis
    - pca: PCA object used to project data onto principal component axes
        (sklearn.pipeline.Pipeline, can include scaling as pre-processing step)
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
    for model_manifest in model_manifest_list:
        print("**** Generating train/test sets for dataset", model_manifest.dataset_name, "**** \n")

        # load DiffAE feature data from this one dataset
        # and get features projected onto principal component axes
        # as defined by fit PCA object pca.
        df_proj = get_manifest_for_dynamics_workflows(model_manifest, pca=pca)

        # get train test split for this dataset
        x_train, x_test, y_train, y_test, v_train, v_test, u_train, u_test = (
            kramers_moyal_train_test_one_dataset(
                df_proj,
                model_manifest.dataset_name,
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
