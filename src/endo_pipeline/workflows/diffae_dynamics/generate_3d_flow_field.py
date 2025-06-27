import fire
import numpy as np

from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import dynamics_io
from src.endo_pipeline.configs.dataset_config import (
    DatasetConfig,
    load_reference_dataset_configs,
    load_single_dataset_config,
)
from src.endo_pipeline.library.analyze.diffae_manifest import manifest_pca, preprocessing
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.diffae_features import manifest_viz


def main(list_of_datasets: list[str] | None = None) -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the
    DiffAE crop-based features for each of the single flow datasets.
    """
    # Create output folder if does not exist yet
    workflow_fig_folder = "flow_field_3d/figs"
    workflow_output_folder = "flow_field_3d/outputs"
    workflow_vtk_folder = "flow_field_3d/outputs/vtks"
    output_savedir = get_output_path(workflow_output_folder, verbose=False)
    fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
    vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

    # if not provided in command line, run
    # on default list of datasets
    if list_of_datasets is None:
        list_of_datasets = [
            "20241120_20X",
            "20250409_20X",
            "20241217_20X",
            "20250428_20X",
            "20250319_20X",
            "20250326_20X",
        ]

    datasets_to_use = [
        load_single_dataset_config(dataset_name) for dataset_name in list_of_datasets
    ]

    pca = manifest_pca.fit_pca()

    # plot scatter of PCA components
    # for a) just the datasets used to fit PCA
    # and b) all datasets specified in the command line
    #   (or default list, if not specified)
    # get timepoints to use for scatter plots
    # all timepoints except no flow
    pca_ref_configs = load_reference_dataset_configs()
    restrict_no_flow = True  # restrict plot to subset of no flow timepoints

    # get timepoints to use for scatter plots
    # this can definitely be written into a wrapper function
    # maybe make a dictionary instead of a list?
    timepoints_refs = preprocessing.get_timepoints_for_plotting_pcs(
        pca_ref_configs, restrict_no_flow=restrict_no_flow
    )

    # scatter plot of pca reference datasets
    fig, _ = manifest_viz.plot_pc_scatter(
        pca, pca_ref_configs, timepoints_to_use=timepoints_refs  # pca reference datasets
    )
    viz_base.save_plot(fig, fig_savedir + "/pca_scatter_ref")

    # scatter plot of all datasets specified in command line
    fig, _ = manifest_viz.plot_pc_scatter(
        pca,
        datasets_to_use,  # all datasets specified and all timepoints
    )
    viz_base.save_plot(fig, fig_savedir + "/pca_scatter_all")

    # load default config, get kernel params
    dynamics_config = dynamics_io.load_dynamics_config("default")
    kernel_params = dynamics_config["kramers_moyal"]["kernel_params"]

    # get time between frames
    # in minutes
    dt = dynamics_config["dt"]

    # time span for the ODE solver
    # units for time steps are in minutes
    # 48 hours in minutes =
    # 48 * 60 = 2880 time steps
    time_span = [0, 2880]

    # initial condition for the ODE solver
    # this is fixed across datasets /
    # shear stress conditions
    init = np.array([-0.1, -0.7, -0.1])

    data_driven_flow_field.ddff_main(
        datasets_to_use,
        pca,
        kernel_params,
        dt,
        time_span,
        init,
        fig_savedir,
        vtk_savedir,
        output_savedir,
    )


if __name__ == "__main__":
    fire.Fire(main)
