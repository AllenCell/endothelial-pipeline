TAGS = ["dynamical_systems", "diffae_features"]


def main(dataset_name: str = "3d_flow_field_analysis", model_name: str = "diffae_04_10") -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for each of the single flow datasets.

    Parameters
    ----------
    dataset_name
        Dataset(s) to load images from, either a single dataset name or the name
        of a dataset collection.
    model_name
        Name of the model for which to load the feature manifest data.

    Returns
    -------
    :
        Saves the PCA scatter plots, flow field analysis results, and visualizations
        to the specified output directories.
    """
    import logging

    import numpy as np

    from endo_pipeline.configs import (
        dynamics_io,
        get_available_dataset_collection_names,
        get_available_dataset_names,
        get_datasets_in_collection,
    )
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_features import get_and_analyze_ddff
    from endo_pipeline.library.analyze.diffae_manifest import fit_pca
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.manifests import load_dataframe_manifest

    logger = logging.getLogger(__name__)

    # Create output folder if does not exist yet
    workflow_name = "flow_field_3d"
    output_savedir = get_output_path(workflow_name, model_name, "outputs", include_timestamp=False)
    fig_savedir = get_output_path(workflow_name, model_name, "figs", include_timestamp=False)
    vtk_savedir = get_output_path(
        workflow_name, model_name, "outputs", "vtk", include_timestamp=False
    )

    manifest = load_dataframe_manifest(model_name)

    # check if input is a dataset collection or a single dataset name
    if dataset_name in get_available_dataset_collection_names():
        # if it is a dataset collection, load all datasets in the collection
        dataset_names = get_datasets_in_collection(dataset_name, list(manifest.locations.keys()))
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

    # plot scatter of PCA components and all datasets specified in the command
    # line (or default list, if not specified)
    fig, _ = feature_viz.plot_pc_scatter(dataset_names, manifest, pca)
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
        dataset_names,
        manifest,
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
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
