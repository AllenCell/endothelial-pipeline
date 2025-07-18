import fire

from src.endo_pipeline.configs import dynamics_io, get_timelapse_model_manifests, load_model_config
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_features import (
    build_kramers_moyal_train_test,
    save_train_test,
)
from src.endo_pipeline.library.analyze.diffae_manifest import fit_pca
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.diffae_features import manifest_viz


def main(dynamics_config_name: str = "default", model_name: str = "diffae_04_10") -> None:
    """
    Build training and test data for regression
    model fitting and evaluation of the dynamical
    systems model for the manifest data (Diff AE).

    Input:
    - dynamics_config_name (str): Name of the configuration to load from dynamics_config.yaml.
        Default is "default".
    - model_name (str): Name of the model to load from model_config.yaml.
        Analysis will be performed on the model manifest datasets for this model.

    Output:
    - Saves the training and test data for regression model
        fitting in a specified directory. Saved out as a
        dictionary with keys "X_train", "X_test", "Y_train", "Y_test",
        "V_train", "V_test", "u_train", "u_test",
        where the values Y and V are the estimated
        drift and diffusion terms, respectively, at the points X
        and shear stress u.
    """
    ################### Load dynamics config and fit PCA ###################
    # make save directory for workflow outputs (set in config file dynamics_config.yaml)
    print("\n", "*** Running workflow using config: ", dynamics_config_name, "\n")
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

    ################### Visualize PCA results ###################
    # plot explained variance ratio of PCA components
    fig, _ = manifest_viz.plot_explained_variance(pca["pca"].explained_variance_ratio_)
    viz_base.save_plot(
        fig, filename=fig_savedir / "explained_variance_ratio", format=".png", dpi=500
    )

    ################### Build train-test data for regression ###################
    # load inputs from dynamics_config.yaml
    pcs = dynamics_config["pcs_to_analyze"]
    dt = dynamics_config["dt"]
    kramers_moyal_config = dynamics_config["kramers_moyal"]
    num_bins = kramers_moyal_config["num_bins"]
    kernel_params = None
    if "kernel_params" in kramers_moyal_config:
        kernel_params = kramers_moyal_config["kernel_params"]

    # get model config from model name
    model_config = load_model_config(model_name)

    # filter out datasets that are not timelapse
    # and load model manifests
    model_manifest_list = get_timelapse_model_manifests(model_config)

    # build train-test data for regression
    train_test_dict = build_kramers_moyal_train_test(
        model_manifest_list,
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
    fire.Fire(main)
