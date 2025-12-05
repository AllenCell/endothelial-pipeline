from endo_pipeline.cli import Datasets
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["dynamical_systems", "diffae_features"]


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
) -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for each of the single flow datasets.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to use for visualization.
    model_manifest_name
        Name of the model manifest containing the run to load features from.
    run_name
        Name of the specific model run to load featuref for. If None, uses the most recent run.

    Returns
    -------
    :
        Saves the PCA scatter plots, flow field analysis results, and visualizations
        to the specified output directories.
    """

    import numpy as np

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.analyze.dynamics_utils import get_and_analyze_ddff
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.flow_field_3d import (
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        INIT_POINT_3D,
        KERNEL_PARAMS_3D,
        NUM_BINS_3D,
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS,
        TIME_STEP_IN_MINUTES,
        TRAJECTORY_TIME_SPAN,
    )

    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )

    # Create output folder if does not exist yet
    output_savedir = get_output_path(
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS,
        dataframe_manifest_name,
        "outputs",
        include_timestamp=False,
    )
    fig_savedir = get_output_path(
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS, dataframe_manifest_name, "figs"
    )
    vtk_savedir = get_output_path(
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS, dataframe_manifest_name, "outputs", "vtk"
    )

    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Default list of datasets if not provided. Filter by datasets available in
    # the manifest.
    valid_dataset_options = list(dataframe_manifest.locations.keys())
    if datasets is None:
        dataset_names = get_datasets_in_collection(
            DATASET_COLLECTION_FOR_3D_DYNAMICS, valid_dataset_options
        )
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]

    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name)

    # plot scatter of PCA components and all datasets specified in the command
    # line (or default list, if not specified)
    fig, _ = feature_viz.plot_pc_scatter(dataset_names, dataframe_manifest, pca)
    save_plot_to_path(fig, fig_savedir, "pca_scatter_all")

    get_and_analyze_ddff(
        dataset_names,
        dataframe_manifest,
        pca,
        kernel_params=KERNEL_PARAMS_3D,
        dt=TIME_STEP_IN_MINUTES,
        time_span=TRAJECTORY_TIME_SPAN,
        init=np.array(INIT_POINT_3D),
        num_bins=NUM_BINS_3D,
        fig_savedir=fig_savedir,
        vtk_savedir=vtk_savedir,
        output_savedir=output_savedir,
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
