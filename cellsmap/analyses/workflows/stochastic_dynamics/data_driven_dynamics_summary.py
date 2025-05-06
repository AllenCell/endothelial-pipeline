import fire
import matplotlib.pyplot as plt
import numpy as np

from cellsmap.analyses.utils import model_analysis
from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils.numerics import data_driven_3D_flow_field as ddff
from cellsmap.analyses.utils.viz import manifest_viz
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import manifest_io
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)
from cellsmap.util.manifest_preprocessing import manifest_pca
from cellsmap.util.set_output import get_output_path


def main(config_name: str = "default") -> None:
    ################### Load manifest data and fit PCA ###################
    # make save directory for workflow outputs (set in config file dynamics_config.yaml)
    print("\n", "*** Running workflow using config: ", config_name, "\n")
    config = dynamics_io.load_dynamics_config(config_name)

    # get output subdirectory for intermediate workflow outputs (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_output_folder = "stochastic_dynamics/" + config["name"] + "/outputs"
    savedir = get_output_path(workflow_output_folder)

    # get output subdirectory for figures that workflow outputs (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_fig_folder = "stochastic_dynamics/" + config["name"] + "/figs"
    fig_savedir = get_output_path(workflow_fig_folder)

    # fit PCA to reference timepoints of reference datasets (removing outliers)
    pca = manifest_pca.fit_pca()

    # save out PCA object (need later for analysis and summary of fit dynamical systems model)
    manifest_io.save_pca_model(pca, savedir)

    ################### Visualize PCA results ###################
    # plot explained variance ratio of PCA components
    fig, _ = manifest_viz.plot_explained_variance(pca["pca"].explained_variance_ratio_)
    vb.save_plot(
        fig, filename=fig_savedir + "explained_variance_ratio", format=".png", dpi=500
    )

    # plot top 3 principal components of feature data vs. frame number
    fig, _ = manifest_viz.plot_top_3_PCs_alldata(pca)
    vb.save_plot(fig, filename=fig_savedir + "top_3_PCs", format=".png", dpi=500)

    ################### Build train-test data for regression ###################
    # load inputs from dynamics_config.yaml
    PCs = config["PCs_to_analyze"]
    dt = config["dt"]
    ds_to_skip = config["datasets_to_skip"]
    kramers_moyal_config = config["kramers_moyal"]
    Nbins = kramers_moyal_config["Nbins"]
    km_method = kramers_moyal_config["method"]
    kernel_params = None
    if "kernel_params" in kramers_moyal_config:
        kernel_params = kramers_moyal_config["kernel_params"]

    pplane_xlim = config["plt_xlim"]["pplane"]
    pplane_ylim = config["plt_ylim"]["pplane"]
    bin_xlim = config["plt_xlim"]["hist"]
    bin_ylim = config["plt_ylim"]["hist"]
    N_pts_pplane = config["N_pts_pplane"]
    N_bins_hist = config["N_bins_hist"]

    # for phase plane plots, fix grid across all datasets
    pplane_xvec = np.linspace(pplane_xlim[0], pplane_xlim[1], N_pts_pplane + 1)
    pplane_yvec = np.linspace(pplane_ylim[0], pplane_ylim[1], N_pts_pplane + 1)

    # for histogram plots, fix bins across all datasets
    bins, centers = rh.get_bins(N_bins_hist, bin_limits=[bin_xlim, bin_ylim])

    # loop through datasets, get flow field estimates, and save out figures
    list_of_datasets = manifest_io.list_datasets_with_manifest("diffae_manifest_fmsid")

    for name in list_of_datasets:
        if name in ds_to_skip:
            print(f"**** Skipping dataset {name}, **** \n")
            continue
        print(f"Computing drift and diffusion fields for dataset {name}")
        df_proj = diffae_preproc.get_manifest_for_dynamics_workflows(name, pca)

        feat_cols_all = manifest_io.get_feature_cols(df_proj)
        feat_cols = [feat_cols_all[i] for i in PCs]  # just get PCs of interest

        # split out data by flow condition
        df_by_flow, shear_list = rh.get_X_by_flow(df_proj, name)
        num_flow = len(shear_list)

        f_KM = []
        D_KM = []

        for j in range(num_flow):
            # get list of per-crop trajectories, the corresponding displacement vectors, and time differences
            X_list, dX_list, dT_list = rh.get_X_dX_and_dT(
                df_by_flow[j], feat_cols=feat_cols
            )

            # get bins for histogramming (for drift and diffusion estimates)
            # bins, centers = rh.get_bins(Nbins, data=X_list)

            # get drift and diffusion estimates (Kramers-Moyal coefficients)
            f_KM, D_KM = rh.get_kramers_moyal(
                X_list,
                dX_list,
                dT_list,
                bins,
                dt,
                method=km_method,
                kernel_params=kernel_params,
            )

            # extrapolate nans in drift and diffusion estimates
            drift_dict = ddff.compute_extrapolated_vector_field(
                f_KM, centers, interpolator="nearest"
            )
            diffusion_dict = ddff.compute_extrapolated_vector_field(
                D_KM, centers, interpolator="nearest"
            )

            drift = ddff.get_callable_vector_field(drift_dict, for_solve_ivp=False)
            drift_ = lambda x, u: drift(x)
            diffusion = ddff.get_callable_vector_field(
                diffusion_dict, for_solve_ivp=False
            )
            diffusion_ = lambda x, u: diffusion(x)

            fig1, _, fig2, _ = model_analysis.model_data_comparison_one_dataset(
                [drift_, diffusion_],
                df_proj,
                shear_list[j],
                PCs,
                bins,
                pplane_xvec,
                pplane_yvec,
                use_fipy=False,
            )

            sup_title = fig2._suptitle.get_text()
            sup_title = name + ", " + str(shear_list[j]) + " dyn/cm$^2$ \n" + sup_title
            fig2.suptitle(sup_title, fontsize=fig2._suptitle.get_fontsize(), y=1.15)

            # save figures
            vb.save_plot(
                fig1,
                fig_savedir
                + name
                + "_ddff_phase_portrait_shear_"
                + str(int(shear_list[j])),
            )
            vb.save_plot(
                fig2,
                fig_savedir
                + name
                + "_ddff_stationary_dist_shear_"
                + str(int(shear_list[j])),
            )


if __name__ == "__main__":
    fire.Fire(main)
