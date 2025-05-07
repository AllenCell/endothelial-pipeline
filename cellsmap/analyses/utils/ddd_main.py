import numpy as np

from cellsmap.analyses.utils import model_analysis, regression_helper as rh
from cellsmap.analyses.utils.numerics import data_driven_flow_field as ddff
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)
from cellsmap.util import manifest_io

def ddd_model_analysis(
        name, model, data, shear, pcs, bins, fig_savedir, config
) -> None:
    # for phase plane plots, grid is fixed and
    # set in the config file dynamics_config.yaml
    pplane_xlim = config["plt_xlim"]["pplane"]
    pplane_ylim = config["plt_ylim"]["pplane"]
    num_pts_pplane = config["num_pts_pplane"]

    pplane_xvec = np.linspace(pplane_xlim[0], pplane_xlim[1], num_pts_pplane + 1)
    pplane_yvec = np.linspace(pplane_ylim[0], pplane_ylim[1], num_pts_pplane + 1)

    # call main model analysis function
    fig1, _, fig2, _ = model_analysis.model_data_comparison_one_dataset(
            model,
            data,
            shear,
            pcs,
            bins,
            pplane_xvec,
            pplane_yvec,
        )

    # add title to histogram plots
    sup_title = fig2._suptitle.get_text()
    sup_title = name + f", {shear} dyn/cm$^2$ \n" + sup_title
    fig2.suptitle(sup_title, fontsize=fig2._suptitle.get_fontsize(), y=1.15)

    # save figures
    vb.save_plot(
        fig1,
        fig_savedir
        + name
        + f"_ddff_phase_portrait_shear_{int(shear)}"
    )
    vb.save_plot(
        fig2,
        fig_savedir
        + name
        + f"_ddff_stationary_dist_shear_{int(shear)}"
    )
    return

def get_and_analyze_ddd(
        name,pca,kernel_params,fig_savedir,config
) -> None:
    """
    Get and analyze data-driven dynamics for a given dataset.


    """
    # unpack relevant parameters from config:

    # which PCs to analyze (2D)
    pcs = config["pcs_to_analyze"]
    # time step (in minutes)
    dt = config["dt"]

    # bin limits for ddff and histogram
    # fixed across all datasets
    bin_xlim = config["plt_xlim"]["hist"]
    bin_ylim = config["plt_ylim"]["hist"]
    num_bins_hist = config["num_bins_hist"]
    # get bins edges and centers
    bins, centers = rh.get_bins(num_bins_hist, bin_limits=[bin_xlim, bin_ylim])

    # load the data for the given name
    # and preprocess it
    df_proj = diffae_preproc.get_manifest_for_dynamics_workflows(name, pca)

    # just get PCs of interest
    feat_cols_all = manifest_io.get_feature_cols(df_proj)
    feat_cols = [feat_cols_all[i] for i in pcs]

    # split out data by flow condition
    df_by_flow, shear_list = rh.get_traj_by_flow(df_proj, name)
    num_flow = len(shear_list)

    # getting drift and diffusion estimates
    # for each flow condition
    drift_km = []
    diff_km = []

    for j in range(num_flow):
        # get list of per-crop trajectories and list
        # of the corresponding displacement vectors
        traj_list, d_traj_list = rh.get_traj_and_diff(df_by_flow[j], feat_cols)

        # get drift and diffusion estimates (Kramers-Moyal coefficients)
        drift_km, diff_km = rh.get_kramers_moyal(
            traj_list,
            d_traj_list,
            bins,
            dt,
            kernel_params=kernel_params,
        )

        # extrapolate nans in drift and diffusion estimates
        drift_dict = ddff.compute_extrapolated_vector_field(
            drift_km, centers, interpolator="nearest"
        )
        diffusion_dict = ddff.compute_extrapolated_vector_field(
            diff_km, centers, interpolator="nearest"
        )

        # turn into callable vector fields
        drift = ddff.get_callable_vector_field(drift_dict, for_solve_ivp=False)
        # have to have shear as a parameter
        # (dummy variable)
        drift_ = lambda x, u: drift(x)
        diffusion = ddff.get_callable_vector_field(
            diffusion_dict, for_solve_ivp=False
        )
        diffusion_ = lambda x, u: diffusion(x)

        # call main model analysis function
        # for the data driven dynamics workflow
        ddff_model_analysis(
            name,
            [drift_, diffusion_],
            df_by_flow[j],
            shear_list[j],
            pcs,
            bins,
            fig_savedir,
            config
        )
    return

        