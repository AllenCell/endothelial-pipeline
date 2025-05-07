# %%
import fire
import numpy as np

from cellsmap.analyses.utils import ddd_main
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.util import manifest_io
from cellsmap.util.manifest_preprocessing import manifest_pca
from cellsmap.util.set_output import get_output_path


# %%
def main(config_name: str = "default") -> None:
    """
    Get and visualize data-driven flow fields for
    all datasets in the manifest using various kernel bandwidths.

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
    workflow_output_folder = "stochastic_dynamics/kernel_sweep/outputs"
    savedir = get_output_path(workflow_output_folder)

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path
    # function will create it
    workflow_fig_folder = "stochastic_dynamics/kernel_sweep/figs"
    fig_savedir = get_output_path(workflow_fig_folder)

    # fit PCA to reference timepoints of
    # reference datasets
    pca = manifest_pca.fit_pca()

    # save out PCA object (need later for analysis
    # and summary of fit dynamical systems model)
    manifest_io.save_pca_model(pca, savedir)

    #### Get data driven flow fields (kernel method) ####
    # get list of all datasets
    # with DiffAE manifest data
    list_of_datasets = manifest_io.list_datasets_with_manifest("diffae_manifest_fmsid")

    # load inputs from dynamics_config.yaml
    ds_to_skip = config["datasets_to_skip"]

    # set kernel type
    kernel_type = "gaussian"

    # range of bandwidths to test
    # log scale between 0.05 and 0.5
    logspace_bw = np.logspace(np.log10(0.05), np.log10(0.5), num=10)

    # loop over bandwidths
    for j, bw in enumerate(logspace_bw):
        kernel_params = {
            "kernel": kernel_type,
            "bandwidth": bw,
        }
        print(f"Running analysis for kernel bandwidth {bw:.3f} \n")

        # make save directory for workflow outputs
        # get string of bandwidth rounded
        # to 3 decimal places
        # and get only the decimal part
        bw_str = f"{bw:.3f}".split(".")[1]
        fig_savedir_kernel = get_output_path(fig_savedir + f"bw_{bw_str}/")
        # loop through datasets, get flow field
        # estimates, and save out figures
        for name in list_of_datasets:
            if name in ds_to_skip:
                print(f"**** Skipping dataset {name}, **** \n")
                continue
            print(f"Computing drift and diffusion fields for dataset {name}")

            ddd_main.get_and_analyze_ddd(
                name, pca, kernel_params, fig_savedir_kernel, config
            )


if __name__ == "__main__":
    fire.Fire(main)
