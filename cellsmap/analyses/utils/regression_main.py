import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.viz import manifest_viz as mv
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import manifest_io as mio
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)


def kramers_moyal_train_test_one_dataset(
    df_proj: pd.DataFrame,
    ds_name: str,
    pcs: list,
    num_bins: list,
    dt: float,
    train_frac: float,
    fig_savedir: str,
    method: str = "kernel",
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
    Generate train test sets for Kramers-Moyal coefficients (drift and diffusion estimates) for one dataset.
    This function is called by build_kramers_moyal_train_test in a loop over all datasets in the dataframe.

    Inputs:
    - df_proj: pandas dataframe containing the dataset of interest, projected onto all principal component axes (change of basis, no dimensionality reduction)
    - ds_name: name of the dataset (used to split out data by flow condition, acessed via data_config.yaml)
    - pcs: list of principal component axes to project data onto for Kramers-Moyal analysis (e.g., [0,1] for first two principal components)
    - num_bins: list of number of bins to use for histogramming data to compute the Kramers-Moyal coefficients (conditional averages computed in each bin)
    - dt: time step between data points (used to compute Kramers-Moyal coefficients)
    - train_frac: fraction of data to use for training
    - method: method to use for computing Kramers-Moyal coefficients ('kernel' or 'histogram', default is 'kernel')

    Outputs:
    - X_train: training data for Kramers-Moyal coefficients (drift and diffusion estimates) from the given dataset
    - X_test: test data for Kramers-Moyal coefficients from the given dataset
    - Y_train: training data for drift estimates from the given dataset
    - Y_test: test data for drift estimates from the given dataset
    - V_train: training data for diffusion estimates from the given dataset
    - V_test: test data for diffusion estimates from the given dataset
    - u_train: training flow conditions (shear rates) from the given dataset
    - u_test: test flow conditions from the given dataset
    """

    # for extracting just the axes (specified via PCs) we want from the resulting dataframe
    # e.g., if we are just analyzing the first two principal components, we want to extract columns 'feat_0' and 'feat_1'
    feat_cols_all = mio.get_feature_cols(df_proj)
    feat_cols = [feat_cols_all[i] for i in pcs]
    ndim = len(pcs)

    # split out data by flow condition
    df_by_flow, shear_list = rh.get_X_by_flow(df_proj, ds_name)
    num_flow = len(shear_list)

    f_KM = []
    D_KM = []
    X_pts = []

    for j in range(num_flow):
        # get list of per-crop trajectories, the corresponding displacement vectors, and time differences
        X_list, dX_list, dT_list = rh.get_X_dX_and_dT(
            df_by_flow[j], feat_cols=feat_cols
        )

        # get bins for histogramming (for drift and diffusion estimates)
        bins, centers = rh.get_bins(num_bins, data=X_list)

        # get drift and diffusion estimates (Kramers-Moyal coefficients)
        f_KM_, D_KM_ = rh.get_kramers_moyal(
            X_list,
            dX_list,
            dT_list,
            bins,
            dt,
            method=method,
            kernel_params=kernel_params,
        )

        # plot drift and diffusion estimates
        kmc = np.concatenate([f_KM_, D_KM_], axis=-1).T
        fig = mv.plot_km(centers, kmc, pcs, shear_list[j])[0]
        vb.save_plot(
            fig,
            filename=fig_savedir + f"kmcs_all_{ds_name}_flow_{j}",
            format=".png",
            dpi=500,
        )

        # quiver and streamplot of drift vector field
        if ndim == 2:
            fig = mv.plot_km_drift_2d(centers, kmc, pcs, shear_list[j])[0]
            vb.save_plot(
                fig,
                filename=fig_savedir + f"kmcs_drift_{ds_name}_flow_{j}",
                format=".png",
                dpi=500,
            )

        # remove NaNs from drift and diffusion estimates (bins with no data), get corresponding bin centers as well
        (
            f_KM_noNAN,
            X_pts_,
        ) = rh.masked_vector_field(f_KM_, np.array(np.meshgrid(*centers)).T)
        D_KM_noNAN, _ = rh.masked_vector_field(D_KM_, np.array(np.meshgrid(*centers)).T)
        f_KM.append(f_KM_noNAN)
        D_KM.append(D_KM_noNAN)
        X_pts.append(X_pts_)

    del df_by_flow  # free up memory

    # get train test split of Kramers-Moyal estimates for each flow condition
    X_train, X_test, Y_train, Y_test, V_train, V_test = rh.train_test_all(
        X_pts, f_KM, D_KM, train_frac, seed=47
    )

    # get number of training and test points for each flow condition
    N_tot = [X_pts[j].shape[0] for j in range(num_flow)]
    N_train = [int(train_frac * N_tot[j]) for j in range(num_flow)]
    N_test = [N_tot[j] - N_train[j] for j in range(num_flow)]

    # get corresponding flow condition for each training and test point
    u_train = np.concatenate(
        [shear_list[j] * np.ones((N_train[j], 1)) for j in range(num_flow)]
    )
    u_test = np.concatenate(
        [shear_list[j] * np.ones((N_test[j], 1)) for j in range(num_flow)]
    )

    del X_pts, f_KM, D_KM  # free up memory

    return X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test


def build_kramers_moyal_train_test(
    pca: Pipeline,
    pcs: list[int],
    num_bins: list[int],
    dt: float,
    ds_to_skip: list[str],
    fig_savedir: str,
    train_frac: float = 0.8,
    method: str = "kernel",
    kernel_params: dict | None = None,
) -> dict:
    """
    Build train test sets for Kramers-Moyal coefficients (drift and diffusion estimates) for all datasets in the dataframe df.

    Inputs:
    - df: pandas dataframe containing all datasets (loaded manifest file with DiffAE output)
    - pca: PCA object used to project data onto principal component axes (sklearn.pipeline.Pipeline, can include scaling as pre-processing step)
    - PCs: list of principal component axes to use for Kramers-Moyal analysis
    - Nbins: list of number of bins to use for histogramming data to compute the Kramers-Moyal coefficients (conditional averages computed in each bin)
    - dt: time step between data points (used to compute Kramers-Moyal coefficients)
    - ds_to_skip: list of dataset names to skip when building train test sets (e.g., if a dataset is known to be problematic)
    - train_frac: fraction of data to use for training (default is 0.8)
    - method: method to use for computing Kramers-Moyal coefficients ('kernel' or 'histogram', default is 'kernel')
    - kernel_params: dictionary of parameters for kernel method (default is None, which uses default parameters if method is 'kernel')

    Outputs:
    - out_dict: dictionary containing the following keys:
        - 'X_train': training data for Kramers-Moyal coefficients (drift and diffusion estimates)
        - 'X_test': test data for Kramers-Moyal coefficients
        - 'Y_train': training data for drift estimates
        - 'Y_test': test data for drift estimates
        - 'V_train': training data for diffusion estimates
        - 'V_test': test data for diffusion estimates
        - 'u_train': training flow conditions (shear rates) - passed in as control variable
        - 'u_test': test flow conditions
    The train test sets are concatenated across all datasets in the dataframe.
    """
    # get list of datasets with DiffAE manifest data
    list_of_datasets = mio.list_datasets_with_manifest("diffae_manifest_fmsid")

    # initialize lists to store train test sets for each dataset
    X_train_list = []
    X_test_list = []
    Y_train_list = []
    Y_test_list = []
    V_train_list = []
    V_test_list = []
    u_train_list = []
    u_test_list = []

    # for each dataset, generate train test sets for drift and diffusion estimates
    # (Kramers-Moyal coefficients, Y and V, respectively)
    for ds_name in list_of_datasets:
        # skip specified datasets when building train test sets
        if ds_name in ds_to_skip:
            print("**** Skipping dataset", ds_name, "**** \n")
            continue

        print("**** Generating train/test sets for dataset", ds_name, "**** \n")

        # load DiffAE feature data from this one dataset, with outliers labeled and features
        # projected onto principal component axes as defined by fit PCA object pca
        df_proj = diffae_preproc.get_manifest_for_dynamics_workflows(ds_name, pca)

        # get train test split for this dataset
        X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test = (
            kramers_moyal_train_test_one_dataset(
                df_proj,
                ds_name,
                pcs,
                num_bins,
                dt,
                train_frac,
                fig_savedir,
                method=method,
                kernel_params=kernel_params,
            )
        )

        # add train test for this dataset to list
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        Y_train_list.append(Y_train)
        Y_test_list.append(Y_test)
        V_train_list.append(V_train)
        V_test_list.append(V_test)
        u_train_list.append(u_train)
        u_test_list.append(u_test)

        del (
            X_train,
            X_test,
            Y_train,
            Y_test,
            V_train,
            V_test,
            u_train,
            u_test,
        )  # free up memory

    # concatenate all per-dataset train test sets to get final train test sets
    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    Y_train = np.concatenate(Y_train_list)
    Y_test = np.concatenate(Y_test_list)
    V_train = np.concatenate(V_train_list)
    V_test = np.concatenate(V_test_list)
    u_train = np.concatenate(u_train_list)
    u_test = np.concatenate(u_test_list)

    # store final train test sets in dictionary
    out_dict = {
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "V_train": V_train,
        "V_test": V_test,
        "u_train": u_train,
        "u_test": u_test,
    }

    return out_dict
