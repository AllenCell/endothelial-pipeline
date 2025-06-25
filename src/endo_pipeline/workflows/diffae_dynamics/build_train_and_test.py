import fire

from cellsmap.util import manifest_io_temp
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import dynamics_io
from src.endo_pipeline.library.analyze.diffae_features import regression_main
from src.endo_pipeline.library.analyze.diffae_manifest import manifest_pca
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.diffae_features import manifest_viz


def main(config_name: str = "default") -> None:
    """
    Build training and test data for regression
    model fitting and evaluation of the dynamical
    systems model for the manifest data (Diff AE).

    Input:
    - config_name (str): Name of the configuration to load from dynamics_config.yaml.
        Default is "default".

    Output:
    - Saves the training and test data for regression model
        fitting in a specified directory. Saved out as a
        dictionary with keys "X_train", "X_test", "Y_train", "Y_test",
        "V_train", "V_test", "u_train", "u_test",
        where the values Y and V are the estimated
        drift and diffusion terms, respectively, at the points X
        and shear stress u.
    """
    ################### Load manifest data and fit PCA ###################
    # make save directory for workflow outputs (set in config file dynamics_config.yaml)
    print("\n", "*** Running workflow using config: ", config_name, "\n")
    config = dynamics_io.load_dynamics_config(config_name)

    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_output_folder = "stochastic_dynamics/" + config["name"] + "/outputs"
    savedir = get_output_path(workflow_output_folder)

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_fig_folder = "stochastic_dynamics/" + config["name"] + "/figs"
    fig_savedir = get_output_path(workflow_fig_folder)

    # fit PCA to reference timepoints of reference datasets
    pca = manifest_pca.fit_pca()

    # save out PCA object (need later for analysis and
    # summary of fit dynamical systems model)
    manifest_io_temp.save_pca_model(pca, savedir)

    ################### Visualize PCA results ###################
    # plot explained variance ratio of PCA components
    fig, _ = manifest_viz.plot_explained_variance(pca["pca"].explained_variance_ratio_)
    viz_base.save_plot(
        fig, filename=fig_savedir + "explained_variance_ratio", format=".png", dpi=500
    )

    ################### Build train-test data for regression ###################
    # load inputs from dynamics_config.yaml
    pcs = config["pcs_to_analyze"]
    dt = config["dt"]
    ds_to_skip = config["datasets_to_skip"]
    kramers_moyal_config = config["kramers_moyal"]
    num_bins = kramers_moyal_config["num_bins"]
    kernel_params = None
    if "kernel_params" in kramers_moyal_config:
        kernel_params = kramers_moyal_config["kernel_params"]

    # build train-test data for regression
    train_test_dict = regression_main.build_kramers_moyal_train_test(
        pca,
        pcs,
        num_bins,
        dt,
        ds_to_skip,
        fig_savedir,
        kernel_params=kernel_params,
    )

    #### Save train-test data ####
    dynamics_io.save_train_test(train_test_dict, savedir)


if __name__ == "__main__":
    fire.Fire(main)
