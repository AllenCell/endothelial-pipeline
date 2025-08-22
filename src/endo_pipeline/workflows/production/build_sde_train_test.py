TAGS = ["dynamical_systems", "diffae_features", "2d_feature_space"]


def main(dynamics_config_name: str = "default", model_name: str = "diffae_04_10") -> None:
    """
    Build train/test sets to apply SINDy to the Kramers-Moyal estimates from the Diff AE features.

    Parameters
    ----------
    dynamics_config_name
        Name of the configuration to load from dynamics_config.yaml.
    model_name
        Name of the model for which to load the feature manifest data.

    Returns
    -------
    :
        Saves the training and test data for regression model fitting in a specified
        directory. Saved out as a dictionary with keys:
        "X_train", "X_test", "Y_train", "Y_test", "V_train", "V_test", "u_train", "u_test".
        The values Y and V are the estimated drift and diffusion terms, respectively,
        at the points X and shear stress u.
    """
    import logging

    from endo_pipeline.configs import dynamics_io, get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_features import (
        build_kramers_moyal_train_test,
        save_train_test,
    )
    from endo_pipeline.library.analyze.diffae_manifest import fit_pca
    from endo_pipeline.manifests import load_dataframe_manifest

    logger = logging.getLogger(__name__)

    ################### Load dynamics config and fit PCA ###################
    # make save directory for workflow outputs (set in config file dynamics_config.yaml)
    logger.info("*** Running workflow using workflow input config: [ %s ]", dynamics_config_name)
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

    # fit PCA to reference timepoints of reference datasets
    pca = fit_pca(model_name=model_name)

    ################### Build train-test data for regression ###################
    # load inputs from dynamics_config.yaml
    pcs = dynamics_config["pcs_to_analyze"]
    dt = dynamics_config["dt"]
    kramers_moyal_config = dynamics_config["kramers_moyal"]
    num_bins = kramers_moyal_config["num_bins"]
    kernel_params = None
    if "kernel_params" in kramers_moyal_config:
        kernel_params = kramers_moyal_config["kernel_params"]

    # load dataset collection and dataframe manifest for model
    manifest = load_dataframe_manifest(model_name)
    dataset_names = get_datasets_in_collection("timelapse", list(manifest.locations.keys()))

    # build train-test data for regression
    train_test_dict = build_kramers_moyal_train_test(
        dataset_names,
        manifest,
        pca,
        pcs,
        num_bins,
        dt,
        fig_savedir,
        kernel_params=kernel_params,
    )

    #### Save train-test data ####
    save_train_test(train_test_dict, savedir)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
