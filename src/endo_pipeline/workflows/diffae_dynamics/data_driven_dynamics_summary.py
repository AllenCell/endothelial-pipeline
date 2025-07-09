import fire

from src.endo_pipeline.configs import dynamics_io, get_timelapse_model_manifests, load_model_config
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_features import ddd_main
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca


def main(dynamics_config_name: str = "default", model_name: str = "diffae_04_10") -> None:
    """
    Get and visualize data-driven flow fields for all datasets in the manifest.

    Includes model summary and comparison to data as used in, e.g.,
    `cellsmap.analyses.workflows.stochastic_dynamics.dynamics_summarize`.
    """
    #### Load manifest data and fit PCA ####
    # make save directory for workflow outputs
    # (set in config file dynamics_config.yaml)
    print("\n", "*** Running workflow using config: ", dynamics_config_name, "\n")
    dynamics_config = dynamics_io.load_dynamics_config(dynamics_config_name)

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    fig_savedir = get_output_path(
        "stochastic_dynamics", dynamics_config_name, model_name, "figs", include_timestamp=False
    )

    # fit PCA to reference timepoints of
    # reference datasets
    pca = fit_pca()

    #### Get data driven flow fields (kernel method) ####
    # load inputs from dynamics_config.yaml
    kramers_moyal_config = dynamics_config["kramers_moyal"]
    kernel_params = None
    if "kernel_params" in kramers_moyal_config:
        kernel_params = kramers_moyal_config["kernel_params"]

    #################### Load model manifest data ###################
    # get model config from model name
    model_config = load_model_config(model_name)

    # filter out datasets that are not timelapse
    # and load model manifests
    model_manifest_list = get_timelapse_model_manifests(model_config)

    for model_manifest in model_manifest_list:
        print(f"Computing drift and diffusion fields for dataset {model_manifest.dataset_name}")

        ddd_main.get_and_analyze_ddd(
            model_manifest,
            pca,
            kernel_params,
            fig_savedir,
            dynamics_config,
        )


if __name__ == "__main__":
    fire.Fire(main)
