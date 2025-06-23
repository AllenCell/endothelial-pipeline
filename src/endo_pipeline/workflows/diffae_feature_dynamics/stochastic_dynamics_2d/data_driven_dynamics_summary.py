import fire

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import dynamics_io
from src.endo_pipeline.library.analyze.diffae_feature_dynamics import ddd_main
from src.endo_pipeline.library.analyze.diffae_manifest import manifest_pca
from src.endo_pipeline.library.visualize import viz_base as vb
from src.endo_pipeline.library.visualize.diffae_feature_dynamics import manifest_viz


def main(config_name: str = "default") -> None:
    """
    Get and visualize data-driven flow fields for all datasets in the manifest.

    Includes model summary and comparison to data as used in, e.g.,
    `cellsmap.analyses.workflows.stochastic_dynamics.dynamics_summarize`.
    """
    #### Load manifest data and fit PCA ####
    # make save directory for workflow outputs
    # (set in config file dynamics_config.yaml)
    print("\n", "*** Running workflow using config: ", config_name, "\n")
    config = dynamics_io.load_dynamics_config(config_name)

    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path
    # function will create it
    workflow_output_folder = "stochastic_dynamics/" + config["name"] + "/outputs"
    savedir = get_output_path(workflow_output_folder)

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path
    # function will create it
    workflow_fig_folder = "stochastic_dynamics/" + config["name"] + "/figs"
    fig_savedir = get_output_path(workflow_fig_folder)

    # fit PCA to reference timepoints of
    # reference datasets
    pca = manifest_pca.fit_pca()

    # save out PCA object (need later for analysis
    # and summary of fit dynamical systems model)
    manifest_io.save_pca_model(pca, savedir)

    #### Visualize PCA results ####
    # plot explained variance ratio of PCA components
    fig, _ = manifest_viz.plot_explained_variance(pca["pca"].explained_variance_ratio_)
    vb.save_plot(fig, filename=fig_savedir + "explained_variance_ratio", format=".png", dpi=500)

    #### Get data driven flow fields (kernel method) ####
    # load inputs from dynamics_config.yaml
    ds_to_skip = config["datasets_to_skip"]
    kramers_moyal_config = config["kramers_moyal"]
    kernel_params = None
    if "kernel_params" in kramers_moyal_config:
        kernel_params = kramers_moyal_config["kernel_params"]

    # loop through datasets, get flow field
    # estimates, and save out figures
    list_of_datasets = manifest_io.list_datasets_with_manifest("diffae_manifest_fmsid")

    for name in list_of_datasets:
        if name in ds_to_skip:
            print(f"**** Skipping dataset {name} **** \n")
            continue
        print(f"Computing drift and diffusion fields for dataset {name}")

        ddd_main.get_and_analyze_ddd(
            name,
            pca,
            kernel_params,
            fig_savedir,
            config,
        )


if __name__ == "__main__":
    fire.Fire(main)
