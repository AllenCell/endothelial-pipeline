import fire
import numpy as np

from src.endo_pipeline.configs import (
    dynamics_io,
    get_model_manifest,
    get_pca_reference_model_manifests,
    load_dataset_config,
    load_model_config,
)
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_manifest import manifest_pca, preprocessing
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.diffae_features import manifest_viz


def main(dataset_names: str | list[str] | None = None, model_name: str = "diffae_04_10") -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the
    DiffAE crop-based features for each of the single flow datasets.
    """
    # Create output folder if does not exist yet
    workflow_name = "flow_field_3d"
    workflow_output_path = get_output_path(workflow_name, model_name, include_timestamp=False)
    output_savedir = workflow_output_path / "outputs"
    fig_savedir = workflow_output_path / "figs"
    vtk_savedir = output_savedir / "vtk"

    if isinstance(dataset_names, str):
        # if a single dataset is provided, convert to list
        dataset_names = [dataset_names]
    elif dataset_names is None:
        # if not provided in command line, run
        # on default list of datasets
        dataset_names = [
            "20241120_20X",
            "20250409_20X",
            "20241217_20X",
            "20250428_20X",
            "20250319_20X",
            "20250326_20X",
        ]
    pca = manifest_pca.fit_pca()

    # plot scatter of PCA components
    # for a) just the datasets used to fit PCA
    # and b) all datasets specified in the command line
    #   (or default list, if not specified)
    # get timepoints to use for scatter plots
    # all timepoints except no flow
    model_config = load_model_config(model_name)
    pca_ref_model_manifest_list = get_pca_reference_model_manifests(model_config)
    # pca_ref_configs = load_reference_dataset_configs()
    pca_ref_configs = [
        load_dataset_config(model_manifest.dataset_name)
        for model_manifest in pca_ref_model_manifest_list
    ]
    restrict_no_flow = True  # restrict plot to subset of no flow timepoints

    # get timepoints to use for scatter plots
    # this can definitely be written into a wrapper function
    # maybe make a dictionary instead of a list?
    timepoints_refs = preprocessing.get_timepoints_for_plotting_pcs(
        pca_ref_configs, restrict_no_flow=restrict_no_flow
    )

    # scatter plot of pca reference datasets
    fig, _ = manifest_viz.plot_pc_scatter(
        pca, pca_ref_model_manifest_list, timepoints_to_use=timepoints_refs
    )
    viz_base.save_plot(fig, fig_savedir / "pca_scatter_ref")

    # scatter plot of all datasets specified in command line
    model_manifest_list = [
        get_model_manifest(dataset_name, model_config) for dataset_name in dataset_names
    ]
    fig, _ = manifest_viz.plot_pc_scatter(
        pca,
        model_manifest_list,  # all datasets specified and all timepoints
    )
    viz_base.save_plot(fig, fig_savedir / "pca_scatter_all")

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
        model_manifest_list,
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
