from endo_pipeline.cli import Datasets
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["dynamical_systems", "diffae_features"]


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    plot_stack: bool = False,
    compute_vtk: bool = True,
    use_same_axes: bool = False,
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
        c. Optionally, 3D stack plots of the flow field visualizations in each of the three
            variables (if ``plot_stack`` is True).
    2. Optionally, VTK files for 3D flow field saved in the `outputs/vtk/` directory
        (if ``compute_vtk`` is True).
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
    compute_vtk
        If true, compute and save VTK files for 3D flow fields.
    use_same_axes
        If true, use the same axis limits for all datasets when plotting flow fields.
    """
    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, make_name_unique
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.analyze.dynamics_utils.data_driven_flow_field import (
        ddff_model_analysis,
    )
    from endo_pipeline.library.analyze.numerics import get_bins, get_bounds_from_data
    from endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
        plot_stable_fixed_points_together,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_PC_COLUMN_NAMES,
        NUM_PCS_TO_ANALYZE,
        ColumnName,
    )
    from endo_pipeline.settings.flow_field_3d import (
        BIN_WIDTH_DEFAULTS,
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        INIT_POINT_3D,
        KERNEL_PARAMS_3D,
        LOWER_PERCENTILE_FOR_STABLE_FP,
        NUM_INIT_SAMPLES,
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS,
        PAD_BINS_FLOAT,
        TIME_STEP_IN_MINUTES,
        TRAJECTORY_TIME_SPAN,
        UPPER_PERCENTILE_FOR_STABLE_FP,
    )

    logger = logging.getLogger(__name__)

    # load model manifest and get corresponding dataframe manifest name
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )

    # Create output folders if they do not exist yet
    output_savedir = get_output_path(
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS,
        dataframe_manifest_name,
        "outputs",
    )
    fig_savedir = get_output_path(
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS, dataframe_manifest_name, "figs"
    )
    vtk_savedir = get_output_path(
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS, dataframe_manifest_name, "outputs", "vtk"
    )

    # load dataframe manifest with model feature for the given model run
    # and model manifest
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

    # fit PCA using the features from the given dataframe manifest
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name)

    # get common bounds for all datasets
    # will be used for flow field plots if use_common_axis_limits is True
    # regardless, gets used below when plotting stable fixed points together
    bounds_for_plots = get_bounds_from_data(dataset_names, dataframe_manifest, pca)

    # initialize dataframe to hold stable fixed points from all datasets
    # with columns for dataset name and 3D PC space coordinates
    stable_fixed_points_df = pd.DataFrame(columns=[ColumnName.DATASET, *DIFFAE_PC_COLUMN_NAMES[:3]])
    for dataset_name in dataset_names:
        # get bins for KMCs
        bounds_for_km = get_bounds_from_data(
            dataset_names=[dataset_name],
            manifest=dataframe_manifest,
            pca=pca,
            pad=PAD_BINS_FLOAT,
        )
        bins, centers = get_bins(BIN_WIDTH_DEFAULTS, bin_limits=bounds_for_km)
        stable_fixed_points = ddff_model_analysis(
            dataset_name,
            dataframe_manifest,
            pca,
            kernel_params=KERNEL_PARAMS_3D,
            dt=TIME_STEP_IN_MINUTES,
            bins=bins,
            centers=centers,
            time_span=TRAJECTORY_TIME_SPAN,
            init_for_traj=np.array(INIT_POINT_3D),
            num_inits_for_root_solver=NUM_INIT_SAMPLES,
            plot_bounds=bounds_for_plots if use_same_axes else bounds_for_km,
            plot_stack=plot_stack,
            compute_vtk_files=compute_vtk,
            fig_savedir=fig_savedir,
            vtk_savedir=vtk_savedir,
            pc_column_names=DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE],
            lower_percentile=LOWER_PERCENTILE_FOR_STABLE_FP,
            upper_percentile=UPPER_PERCENTILE_FOR_STABLE_FP,
        )

        # add stable fixed points from this dataset to the overall dataframe
        for stable_fp in stable_fixed_points:
            stable_fixed_points_df = pd.concat(
                [
                    stable_fixed_points_df,
                    pd.DataFrame(
                        {
                            ColumnName.DATASET: [dataset_name],
                            DIFFAE_PC_COLUMN_NAMES[0]: [stable_fp[0]],
                            DIFFAE_PC_COLUMN_NAMES[1]: [stable_fp[1]],
                            DIFFAE_PC_COLUMN_NAMES[2]: [stable_fp[2]],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # generate plot of stable fixed points from different datasets overlaid on top of each other
    # (for comparison of stable fixed points across datasets)
    plot_stable_fixed_points_together(stable_fixed_points_df, bounds_for_plots, fig_savedir)

    # save stable fixed points from all datasets to parquet file
    df_file_name = "stable_fixed_points_all_datasets.parquet"
    if DEMO_MODE:
        stable_fixed_points_save_path = make_name_unique(output_savedir / f"demo_{df_file_name}")
    else:
        # eventually, save to FMS
        logger.warning(
            "Saving stable fixed points to FMS not yet implemented, saving locally instead."
        )
        stable_fixed_points_save_path = make_name_unique(output_savedir / df_file_name)

    logger.info(
        "Saving stable fixed points from all datasets to [ %s ]",
        stable_fixed_points_save_path,
    )
    stable_fixed_points_df.to_parquet(
        stable_fixed_points_save_path,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
