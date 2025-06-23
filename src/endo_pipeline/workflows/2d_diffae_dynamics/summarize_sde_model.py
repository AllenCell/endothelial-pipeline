import fire
import numpy as np

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import dynamics_io
from src.endo_pipeline.library.analyze.diffae_feature_dynamics import model_analysis, model_eval
from src.endo_pipeline.library.analyze.diffae_feature_dynamics import regression_helper as rh


def main(config_name: str = "default") -> None:
    """
    Summarize the qualitative and quantitative results of the
    dynamical systems model fit to the manifest data (Diff AE).

    Input:
    - config_name (str): Name of the configuration to load from dynamics_config.yaml.
        Default is "default".

    Output:
    - Saves the figures and analysis results in a specified directory.
        The figures include:
        - Model-data comparison plots (histograms and phase plane plots)
        - Fixed point analysis plots (fixed points as a function of shear stress)
        - Entropy production rate as a function of shear stress
        - Generalized potential energy landscape plots (for various shear stresses)
    """
    ################### Load configs from dynamics_config ###################
    config = dynamics_io.load_dynamics_config(config_name)

    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_output_folder = "stochastic_dynamics/" + config["name"] + "/outputs"
    savedir = get_output_path(workflow_output_folder, verbose=False)

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_fig_folder = "stochastic_dynamics/" + config["name"] + "/figs"
    fig_savedir = get_output_path(workflow_fig_folder, verbose=False)

    # get inputs for analysis/visualization from config
    pcs = config["pcs_to_analyze"]
    ds_to_skip = config["datasets_to_skip"]

    pplane_xlim = config["plt_xlim"]["pplane"]
    pplane_ylim = config["plt_ylim"]["pplane"]
    bin_xlim = config["plt_xlim"]["hist"]
    bin_ylim = config["plt_ylim"]["hist"]
    num_pts_pplane = config["num_pts_pplane"]
    num_bins_hist = config["num_bins_hist"]

    # for phase plane plots, fix grid across all datasets
    pplane_xvec = np.linspace(pplane_xlim[0], pplane_xlim[1], num_pts_pplane + 1)
    pplane_yvec = np.linspace(pplane_ylim[0], pplane_ylim[1], num_pts_pplane + 1)

    # for histogram plots, fix bins across all datasets
    bins, centers = rh.get_bins(num_bins_hist, bin_limits=[bin_xlim, bin_ylim])

    # for plotting fixed points by shear stress
    shear_range = config["shear_range"]
    shear_range_fpt = np.linspace(shear_range[0], shear_range[-1], config["num_shear_fpt"])

    # for plotting entropy production rate by shear stress:
    # check if additive noise or not
    # additive noise: D = const, non-additive noise: D = D(x)
    # this is set in dynamics_config.yaml by setting 'diffusion_feats' to 0 or >0
    additive_noise = True
    if config["polynomial_lib"]["diffusion_feats"] > 0:
        additive_noise = False

    # for plotting generalized potential energy landscape for various shear stresses
    bins_gp, centers_gp = rh.get_bins(config["num_bins_landscape"], bin_limits=[bin_xlim, bin_ylim])
    shear_range_gp = np.linspace(shear_range[0], shear_range[-1], config["num_shear_landscape"])
    downsample_quiver = config["downsample_quiver"]
    normed = config["norm_vectors"]

    ################### Load outputs from dynamics_fit.py ###################
    # load fit drift-diffusion model (list of fit SINDy objects)
    model_dict = dynamics_io.load_model(savedir + "drift_diffusion_model.pkl")

    drift_model = model_dict["drift_model"]
    diff_model = model_dict["diff_model"]

    # convert to callable functions
    drift = model_eval.vector_field_function(drift_model)
    diffusion = model_eval.vector_field_function(diff_model)

    my_model = [drift, diffusion]

    ################### Model-data comparison ###################
    # run comparison of model and data for each dataset
    pca = manifest_io.load_pca_model(savedir)
    model_analysis.model_data_comparison(
        my_model,
        fig_savedir,
        pca,
        pcs,
        bins,
        ds_to_skip,
        pplane_xvec,
        pplane_yvec,
    )

    #### Fixed point analysis ####
    # plot coordinates of fixed points as a function of shear stress
    plt_lims = [
        pplane_xlim,
        pplane_ylim,
    ]  # set limits for plotted/reported fixed points
    model_analysis.run_fixed_point_analysis(drift, shear_range_fpt, pcs, plt_lims, fig_savedir)

    #### Entropy production rate as a function of shear stress ####
    model_analysis.run_epr_analysis(
        my_model, bins, centers, shear_range_fpt, fig_savedir, additive_noise
    )

    #### Generalized potential energy landscape ####

    # plot generalized potential energy landscape for
    # each shear stress specified in shear_range_gp
    model_analysis.run_gen_potential_analysis(
        my_model,
        bins_gp,
        centers_gp,
        shear_range_gp,
        pcs,
        downsample_quiver,
        normed,
        fig_savedir,
        additive_noise,
    )


if __name__ == "__main__":
    fire.Fire(main)
