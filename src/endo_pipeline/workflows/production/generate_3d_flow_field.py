from endo_pipeline.cli import Datasets
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["dynamical_systems", "diffae_features"]


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    plot_stack: bool = False,
) -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for each of the single flow datasets.

    **Flow field estimation and analysis**

    1. Estimate 3D flow fields using a Gaussian kernel method on the PCA-reduced
         DiffAE feature space.
    2. Use interpolation to get a callable flow field function.
    3. Identify stable fixed points in the 3D flow field using a root-finding method
        applied to the flow field function.
    4. Categorize the identified fixed points based on the eigenvalues of the Jacobian
        matrix at each fixed point.
    5. Simulate trajectories in the 3D flow field starting from specified initial points.
    6. Save the flow field analysis results, including stable fixed point locations.

    **Visualization outputs**

    1. 2D flow field visualizations saved as PNG files in the `figs/` directory, including:
        a. 2D slice of the 3D flow field "sliced" according to the coordinates
            of the stable fixed points identified in the 3D flow field.
        b. Trajectories simulated in the 3D flow field, projected onto 2D slices.
    2. VTK files for 3D flow field visualizations saved in the `outputs/vtk/` directory.
    3. Stable fixed point locations from all datasets processed overlaid on a single
        plot saved as a PNG file in the `figs/` directory.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to use for visualization.
    model_manifest_name
        Name of the model manifest containing the run to load features from.
    run_name
        Name of the specific model run to load featuref for. If None, uses the most recent run.
    plot_stack
        If true, plot 3D stacks of the flow field visualizations in each of the three variables.
    """

    import numpy as np

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.analyze.dynamics_utils import get_and_analyze_ddff
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
        NUM_INIT_SAMPLES,
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

    get_and_analyze_ddff(
        dataset_names,
        dataframe_manifest,
        pca,
        kernel_params=KERNEL_PARAMS_3D,
        dt=TIME_STEP_IN_MINUTES,
        time_span=TRAJECTORY_TIME_SPAN,
        init_for_traj=np.array(INIT_POINT_3D),
        num_inits_for_root_solver=NUM_INIT_SAMPLES,
        num_bins=NUM_BINS_3D,
        plot_stack=plot_stack,
        fig_savedir=fig_savedir,
        vtk_savedir=vtk_savedir,
        output_savedir=output_savedir,
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
