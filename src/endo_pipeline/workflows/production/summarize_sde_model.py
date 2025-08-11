TAGS = ["dynamical_systems", "diffae_features"]


def main(dynamics_config_name: str = "default", model_name: str = "diffae_04_10") -> None:
    """
    Summarize the the dynamical systems model fit to the crop-based Diff AE feature data.

    Parameters
    ----------
    dynamics_config_name
        Name of the configuration to load from dynamics_config.yaml.
    model_name
        Name of the model from which manifest data is loaded and analyzed.

    Returns
    -------
    :
        Saves the figures and analysis results in a specified directory.
        The figures include:
        - Model-data comparison plots (histograms and phase plane plots)
        - Fixed point analysis plots (fixed points as a function of shear stress)
        - Entropy production rate as a function of shear stress
        - Generalized potential energy landscape plots (for various shear stresses)
    """

    import numpy as np

    from src.endo_pipeline.configs import dynamics_io, get_datasets_in_collection
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.analyze.diffae_features import (
        load_sde_model,
        model_data_comparison,
        run_epr_analysis,
        run_fixed_point_analysis,
        run_gen_potential_analysis,
    )
    from src.endo_pipeline.library.analyze.diffae_manifest import fit_pca
    from src.endo_pipeline.library.analyze.numerics import get_bins, vector_field_function
    from src.endo_pipeline.manifests import load_dataframe_manifest

    ################### Load configs from dynamics_config ###################
    dynamics_config = dynamics_io.load_dynamics_config(dynamics_config_name)

    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    savedir = get_output_path(
        "stochastic_dynamics", dynamics_config_name, model_name, "outputs", include_timestamp=False
    )

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    fig_savedir = get_output_path(
        "stochastic_dynamics", dynamics_config_name, model_name, "figs", include_timestamp=False
    )

    # get inputs for analysis/visualization from config
    pcs = dynamics_config["pcs_to_analyze"]

    pplane_xlim = dynamics_config["plt_xlim"]["pplane"]
    pplane_ylim = dynamics_config["plt_ylim"]["pplane"]
    bin_xlim = dynamics_config["plt_xlim"]["hist"]
    bin_ylim = dynamics_config["plt_ylim"]["hist"]
    num_pts_pplane = dynamics_config["num_pts_pplane"]
    num_bins_hist = dynamics_config["num_bins_hist"]

    # for phase plane plots, fix grid across all datasets
    pplane_xvec = np.linspace(pplane_xlim[0], pplane_xlim[1], num_pts_pplane + 1)
    pplane_yvec = np.linspace(pplane_ylim[0], pplane_ylim[1], num_pts_pplane + 1)

    # for histogram plots, fix bins across all datasets
    bins, centers = get_bins(num_bins_hist, bin_limits=[bin_xlim, bin_ylim])

    # for plotting fixed points by shear stress
    shear_range = dynamics_config["shear_range"]
    shear_range_fpt = np.linspace(shear_range[0], shear_range[-1], dynamics_config["num_shear_fpt"])

    # for plotting entropy production rate by shear stress:
    # check if additive noise or not
    # additive noise: D = const, non-additive noise: D = D(x)
    # this is set in dynamics_config.yaml by setting 'diffusion_feats' to 0 or >0
    additive_noise = True
    if dynamics_config["polynomial_lib"]["diffusion_feats"] > 0:
        additive_noise = False

    # for plotting generalized potential energy landscape for various shear stresses
    bins_gp, centers_gp = get_bins(
        dynamics_config["num_bins_landscape"], bin_limits=[bin_xlim, bin_ylim]
    )
    shear_range_gp = np.linspace(
        shear_range[0], shear_range[-1], dynamics_config["num_shear_landscape"]
    )
    downsample_quiver = dynamics_config["downsample_quiver"]
    normed = dynamics_config["norm_vectors"]

    ################### Load outputs from dynamics_fit.py ###################
    # load fit drift-diffusion model (list of fit SINDy objects)
    sde_model_dict = load_sde_model(savedir / "drift_diffusion_model.pkl")

    drift_model = sde_model_dict["drift_model"]
    diff_model = sde_model_dict["diff_model"]

    # convert to callable functions
    drift = vector_field_function(drift_model)
    diffusion = vector_field_function(diff_model)

    sde_model = [drift, diffusion]

    #################### Load model manifest data ###################
    manifest = load_dataframe_manifest(model_name)
    dataset_names = get_datasets_in_collection("timelapse", list(manifest.locations.keys()))

    ################### Model-data comparison ###################
    # run comparison of model and data for each dataset
    pca = fit_pca(model_name=model_name)
    model_data_comparison(
        sde_model,
        dataset_names,
        manifest,
        pca,
        pcs,
        bins,
        pplane_xvec,
        pplane_yvec,
        fig_savedir,
    )

    #### Fixed point analysis ####
    # plot coordinates of fixed points as a function of shear stress
    plt_lims = [
        pplane_xlim,
        pplane_ylim,
    ]  # set limits for plotted/reported fixed points
    run_fixed_point_analysis(drift, shear_range_fpt, pcs, plt_lims, fig_savedir)

    #### Entropy production rate as a function of shear stress ####
    run_epr_analysis(sde_model, bins, centers, shear_range_fpt, fig_savedir, additive_noise)

    #### Generalized potential energy landscape ####

    # plot generalized potential energy landscape for
    # each shear stress specified in shear_range_gp
    run_gen_potential_analysis(
        sde_model,
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
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
