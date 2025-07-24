TAGS = ["dynamical_systems", "diffae_features"]


def main(dataset_name: str = "3d_flow_field_analysis", model_name: str = "diffae_04_10") -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for each of the single flow datasets.

    Parameters
    ----------
    dataset_name
        Dataset(s) to apply the model to.
        It should either be a single dataset name or the name of a dataset collection.
        Default is "3d_flow_field_analysis", which is a collection of live, timelapse,
        20X objective, 3i microscope datasets with single flow conditions for each dataset.
    model_name
        Name of the model to load from configs/models/.
        Analysis will be performed on the model manifest datasets for this model.

    Returns
    -------
    None
        Saves the PCA scatter plots, flow field analysis results, and visualizations
        to the specified output directories.

    Raises
    ------
    ValueError
        If the provided dataset name is not a valid dataset or dataset collection name.
    """
    import logging

    import numpy as np

    from src.endo_pipeline.configs import (
        dynamics_io,
        get_available_dataset_collection_names,
        get_available_dataset_names,
        get_datasets_in_collection,
        get_model_manifest,
        get_pca_reference_model_manifests,
        load_dataset_config,
        load_model_config,
    )
    from src.endo_pipeline.io import get_output_path, save_plot_to_path
    from src.endo_pipeline.library.analyze.diffae_features import get_and_analyze_ddff
    from src.endo_pipeline.library.analyze.diffae_manifest import (
        fit_pca,
        get_timepoints_for_plotting_pcs,
    )
    from src.endo_pipeline.library.visualize.diffae_features import manifest_viz

    logger = logging.getLogger(__name__)

    # Create output folder if does not exist yet
    workflow_name = "flow_field_3d"
    output_savedir = get_output_path(workflow_name, model_name, "outputs", include_timestamp=False)
    fig_savedir = get_output_path(workflow_name, model_name, "figs", include_timestamp=False)
    vtk_savedir = get_output_path(
        workflow_name, model_name, "outputs", "vtk", include_timestamp=False
    )

    # check if input is a dataset collection or a single dataset name
    if dataset_name in get_available_dataset_collection_names():
        # if it is a dataset collection, load all datasets in the collection
        dataset_names = get_datasets_in_collection(dataset_name)
    elif dataset_name in get_available_dataset_names():
        # if it is a single dataset name, keep it as is
        dataset_names = [dataset_name]
    else:
        logger.error(
            "Dataset name [ %s ] is not a valid dataset or dataset collection name",
            dataset_name,
        )
        raise ValueError(
            f"Dataset name [ {dataset_name} ] is not a valid",
            "dataset or dataset collection name.",
        )
    pca = fit_pca(model_name=model_name)

    # plot scatter of PCA components
    # for a) just the datasets used to fit PCA
    # and b) all datasets specified in the command line
    #   (or default list, if not specified)
    # get timepoints to use for scatter plots
    # all timepoints except no flow
    model_config = load_model_config(model_name)
    pca_ref_model_manifest_list = get_pca_reference_model_manifests(model_config)
    pca_ref_configs = [
        load_dataset_config(model_manifest.dataset_name)
        for model_manifest in pca_ref_model_manifest_list
    ]
    restrict_no_flow = True  # restrict plot to subset of no flow timepoints

    # get timepoints to use for scatter plots
    # this can definitely be written into a wrapper function
    # maybe make a dictionary instead of a list?
    timepoints_refs = get_timepoints_for_plotting_pcs(
        pca_ref_configs, restrict_no_flow=restrict_no_flow
    )

    # scatter plot of pca reference datasets
    fig, _ = manifest_viz.plot_pc_scatter(
        pca, pca_ref_model_manifest_list, timepoints_to_use=timepoints_refs
    )
    save_plot_to_path(fig, fig_savedir, "pca_scatter_ref")

    # scatter plot of all datasets specified in command line
    model_manifest_list = [
        get_model_manifest(dataset_name, model_config) for dataset_name in dataset_names
    ]
    fig, _ = manifest_viz.plot_pc_scatter(
        pca,
        model_manifest_list,  # all datasets specified and all timepoints
    )
    save_plot_to_path(fig, fig_savedir, "pca_scatter_all")

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

    get_and_analyze_ddff(
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
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
