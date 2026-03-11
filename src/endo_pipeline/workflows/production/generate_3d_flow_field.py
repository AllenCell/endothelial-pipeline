from endo_pipeline.cli import CropPattern, Datasets, StrList
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
    columns: StrList | None = None,
) -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for each of the single flow datasets.

    #dynamical-systems #diffae-feature-analysis

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

    - Stable fixed point locations from all datasets processed overlaid on a single
        plot saved as a PNG file in the `figs/` directory.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to use for visualization.
    model_manifest_name
        Name of the model manifest containing the run to load features from.
    run_name
        Name of the specific model run to load featuref for. If None, uses the most recent run.
    crop_pattern
        The crop pattern to get features for, either "grid" or "tracked".
    columns
        List of column names in the dataframe to use for flow field analysis. If None,
        uses default column names specified in settings.
    """
    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, make_name_unique
    from endo_pipeline.library.analyze.data_driven_flow_field import get_stable_fixed_points
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_traj_and_diff,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins, get_bounds_from_data
    from endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
        plot_stable_fixed_points_together,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        BIN_WIDTH_DEFAULTS,
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        KERNEL_BANDWIDTH,
        KERNEL_FUNCTION_NAME,
        LOWER_PERCENTILE_FOR_STABLE_FP,
        NUM_INIT_SAMPLES,
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS,
        PAD_BINS_FLOAT,
        TIME_STEP_IN_MINUTES,
        UPPER_PERCENTILE_FOR_STABLE_FP,
    )

    logger = logging.getLogger(__name__)

    # load model manifest and get corresponding dataframe manifest name
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern=crop_pattern
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
    if DEMO_MODE:
        logger.warning(
            "DEMO MODE: Using only the first dataset from the manifest for quick visualization."
        )
        dataset_names = dataset_names[:1]

    # get feature column names to use for flow field analysis
    column_names: list[str] = columns or list(DYNAMICS_COLUMN_NAMES)
    if len(column_names) != 3:
        raise ValueError(
            f"Exactly 3 column names must be provided for 3D flow field analysis, but {len(column_names)} were provided: {column_names}"
        )

    # fit PCA using the features from the given dataframe manifest PCA always
    # fit on the grid-based features, even if the features for flow field
    # analysis are from tracked-based crops, to ensure that the PCA space is the
    # same across analyses
    dataframe_manifest_name_pca = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name_pca)

    # get common bounds for all datasets
    # will be used for flow field plots if use_common_axis_limits is True
    # regardless, gets used below when plotting stable fixed points together
    bounds_for_plots = get_bounds_from_data(
        dataset_names, dataframe_manifest, pca, column_names=column_names
    )

    # initialize dataframe to hold stable fixed points from all datasets
    # with columns for dataset name and 3D PC space coordinates
    stable_fixed_points_df = pd.DataFrame(columns=[ColumnName.DATASET, *column_names])

    # initialize kernels and bin widths for each of the three variables for flow
    # field estimation
    kernels = []
    bin_widths = []
    rescaled_theta = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)

    for index, column_name in enumerate(column_names):
        name = KERNEL_NAMES_DYNAMICS.get(column_name, KERNEL_FUNCTION_NAME)
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS.get(column_name, KERNEL_BANDWIDTH)
        period = rescaled_theta if column_name == ColumnName.POLAR_ANGLE else None
        bin_width = BIN_WIDTHS_DYNAMICS.get(column_name, BIN_WIDTH_DEFAULTS[index])

        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
        bin_widths.append(bin_width)

    for dataset_name in dataset_names:
        # get bins for KMCs
        bounds_for_km = get_bounds_from_data(
            dataset_names=[dataset_name],
            manifest=dataframe_manifest,
            pca=pca,
            pad=PAD_BINS_FLOAT,
            column_names=column_names,
        )
        bins, centers = get_bins(bin_widths, bin_limits=bounds_for_km)

        # load dataframe and filter / preprocess it for dynamics workflows (PCA,
        # filter annotated timepoints, transform angular variables),
        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
        )

        # get list of per-crop trajectories, the corresponding
        # displacement vectors, and time differences
        traj_list, d_traj_list = get_traj_and_diff(df, column_names)

        # get drift estimates
        # (Kramers-Moyal coefficients)
        drift_coeffs, _ = get_kramers_moyal_coeffs(
            traj_list, d_traj_list, bins=bins, dt=TIME_STEP_IN_MINUTES, kernel=kernels
        )

        stable_fixed_points = get_stable_fixed_points(
            drift_coeffs=drift_coeffs,
            centers=centers,
            feature_data=df[column_names].to_numpy(),  # get feature data as numpy array
            num_inits_for_root_solver=NUM_INIT_SAMPLES,
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
                            column_names[0]: [stable_fp[0]],
                            column_names[1]: [stable_fp[1]],
                            column_names[2]: [stable_fp[2]],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # generate plot of stable fixed points from different datasets overlaid on top of each other
    # (for comparison of stable fixed points across datasets)
    plot_stable_fixed_points_together(
        stable_fixed_points_df, bounds_for_plots, fig_savedir, column_names
    )

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
