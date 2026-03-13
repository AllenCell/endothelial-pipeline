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
    Generate 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for a given set of datasets.

    #dynamical-systems #diffae-feature-analysis

    **Flow field estimation and analysis**

    1. Estimate 3D flow fields using a Gaussian kernel method on the PCA-reduced
         DiffAE feature space.
    2. Use interpolation to get a callable flow field function.
    3. Identify stable fixed points in the 3D flow field using a root-finding
       method applied to the flow field function.
    4. Save the following outputs locally:
        - Dataframe with the estimated drift coefficients at each grid point for
          each dataset.
        - Dataframe with the corresponding grid point coordinates for each
          dataset.
        - Dataframe with the stable fixed point locations for each dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to use for visualization.
    model_manifest_name
        Name of the model manifest containing the run to load features from.
    run_name
        Name of the specific model run to load featuref for. If None, uses the
        most recent run.
    crop_pattern
        The crop pattern to get features for, either "grid" or "tracked".
    columns
        List of column names in the dataframe to use for flow field analysis. If
        None, uses default column names specified in settings.
    """
    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, make_name_unique
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        get_callable_vector_field,
        get_stable_fixed_points,
    )
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_traj_and_diff,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins, get_bounds_from_data
    from endo_pipeline.manifests import (
        build_dataframe_location_from_path,
        create_dataframe_manifest,
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_THETA_RESCALED,
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        BIN_WIDTH_DEFAULTS,
        DATAFRAME_MANIFEST_PREFIX_DRIFT,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        DATAFRAME_MANIFEST_PREFIX_GRID,
        DATAFRAME_OUTPUT_DIR,
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        KERNEL_BANDWIDTH,
        KERNEL_FUNCTION_NAME,
        LOWER_PERCENTILE_FOR_STABLE_FP,
        NUM_INIT_SAMPLES,
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

    # Create/set output folder for dataframes, save in local directory without
    # timestamp for intermediate level of "static-ness" (ensure they don't get
    # periodically deleted).
    #
    # Also build dataframe manifests for the outputs of this workflow (drift
    # coefficients, grid points, and stable fixed points) with names that
    # include the input dataframe manifest name for traceability and to avoid
    # naming conflicts with other runs. The dataframe manifests get saved to the
    # dataframe manifest directory, and the dataframes themselves get saved to
    # the output directory specified in settings.
    dataframe_savedir = get_output_path(
        DATAFRAME_OUTPUT_DIR, dataframe_manifest_name, include_timestamp=False
    )
    drift_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{dataframe_manifest_name}"
    grid_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_GRID}_{dataframe_manifest_name}"
    fixed_points_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{dataframe_manifest_name}"
    )
    drift_dataframe_manifest = create_dataframe_manifest(
        drift_dataframe_manifest_name, workflow_name=__file__
    )
    grid_dataframe_manifest = create_dataframe_manifest(
        grid_dataframe_manifest_name, workflow_name=__file__
    )
    fixed_points_dataframe_manifest = create_dataframe_manifest(
        fixed_points_dataframe_manifest_name, workflow_name=__file__
    )
    logger.info(
        "Dataframes with 3D flow field estimation results will be saved to: [ %s ]",
        dataframe_savedir,
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
            "DEMO MODE: Using only the first dataset from the manifest for quick testing."
        )
        dataset_names = dataset_names[:1]
        drift_dataframe_manifest_name = f"demo_{drift_dataframe_manifest_name}"
        grid_dataframe_manifest_name = f"demo_{grid_dataframe_manifest_name}"
        fixed_points_dataframe_manifest_name = f"demo_{fixed_points_dataframe_manifest_name}"

    # get feature column names to use for flow field analysis
    column_names: list[str] = columns or list(DYNAMICS_COLUMN_NAMES)
    if len(column_names) != 3:
        raise ValueError(
            f"Exactly 3 column names must be provided for 3D flow field analysis, but {len(column_names)} were provided: {column_names}"
        )
    drift_column_names = [f"{name}_drift" for name in column_names]

    # fit PCA using the features from the given dataframe manifest PCA always
    # fit on the grid-based features, even if the features for flow field
    # analysis are from tracked-based crops, to ensure that the PCA space is the
    # same across analyses
    dataframe_manifest_name_pca = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name_pca)

    # initialize list to hold dataframes of stable fixed points from all
    # datasets with columns for dataset name and 3D PC space coordinates
    stable_fixed_points_all_datasets_list = []

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
        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.shear_stress_regime) > 1:
            logger.warning(
                "Dataset [ %s ] has more than one shear stress condition: [ %s ]. "
                "Skipping for 3D flow field analysis.",
                dataset_name,
                dataset_config.shear_stress_regime,
            )
            continue
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

        # build dataframe with columns for bin centers in each of the three dimensions and
        # the corresponding drift coefficients, to be used for visualization workflow
        drift_coeffs_df = pd.DataFrame(
            {
                ColumnName.DATASET: dataset_name,
                **{
                    drift_column_name: drift_coeffs[..., index].flatten().tolist()
                    for index, drift_column_name in enumerate(drift_column_names)
                },
            }
        )

        # To store as datframe, the grid points are padded with NaN values to
        # ensure that each column has the same number of rows. The grid
        # points will be un-padded in the visualization workflow.
        max_grid_size = max(len(centers[0]), len(centers[1]), len(centers[2]))
        centers_padded = [
            np.pad(
                centers[index],
                (0, max_grid_size - len(centers[index])),
                mode="constant",
                constant_values=np.nan,
            )
            for index in range(len(centers))
        ]
        grid_points_df = pd.DataFrame(
            {
                ColumnName.DATASET: dataset_name,
                **{
                    column_name: centers_padded[index].tolist()
                    for index, column_name in enumerate(column_names)
                },
            }
        )

        # save drift coefficients and grid points dataframes to parquet files,
        # with names that include the input dataframe manifest name for
        # traceability and to avoid naming conflicts with other runs
        drift_coeffs_file_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{dataset_name}.parquet"
        grid_points_file_name = f"{DATAFRAME_MANIFEST_PREFIX_GRID}_{dataset_name}.parquet"
        if DEMO_MODE:
            drift_coeffs_save_path = make_name_unique(
                dataframe_savedir / f"demo_{drift_coeffs_file_name}"
            )
            grid_points_save_path = make_name_unique(
                dataframe_savedir / f"demo_{grid_points_file_name}"
            )
        else:
            # eventually, save to FMS
            logger.warning("Saving dataframes to FMS not yet implemented, saving locally instead.")
            drift_coeffs_save_path = make_name_unique(dataframe_savedir / drift_coeffs_file_name)
            grid_points_save_path = make_name_unique(dataframe_savedir / grid_points_file_name)

        drift_coeffs_df.to_parquet(
            drift_coeffs_save_path,
        )
        grid_points_df.to_parquet(
            grid_points_save_path,
        )

        # add to DataframeManifest for drift coefficients and grid points for this dataset
        drift_location = build_dataframe_location_from_path(drift_coeffs_save_path)
        grid_location = build_dataframe_location_from_path(grid_points_save_path)
        drift_dataframe_manifest.locations[dataset_name] = drift_location
        grid_dataframe_manifest.locations[dataset_name] = grid_location

        ## extrapolate the drift to get a flow field over the entire 3D space as specified by the input bins and centers
        extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
            drift_coeffs, centers, method="linear", for_vtk_files=False
        )

        # get callable drift function to be used for root finding to identify
        # fixed points
        drift_function = get_callable_vector_field(
            extrapolated_flow_field_dict_reg, for_solve_ivp=False, method="linear"
        )

        stable_fixed_points_dataset = get_stable_fixed_points(
            drift_function=drift_function,
            dataframe=df,
            column_names=column_names,
            num_inits_for_root_solver=NUM_INIT_SAMPLES,
            lower_percentile=LOWER_PERCENTILE_FOR_STABLE_FP,
            upper_percentile=UPPER_PERCENTILE_FOR_STABLE_FP,
            polar_angle_range=BIN_LIMITS_THETA_RESCALED if RESCALE_THETA else (-np.pi, np.pi),
        )

        # add stable fixed points from this dataset to the overall dataframe
        # (checking first if returned dataframe is empty first to avoid issues
        # with concatenation and saving an empty dataframe)
        if stable_fixed_points_dataset.empty:
            continue

        stable_fixed_points_all_datasets_list.append(stable_fixed_points_dataset)

        # save stable fixed points from this dataset to parquet file
        stable_fixed_points_file_name = (
            f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{dataset_name}.parquet"
        )
        if DEMO_MODE:
            stable_fixed_points_save_path = make_name_unique(
                dataframe_savedir / f"demo_{stable_fixed_points_file_name}"
            )
        else:
            # eventually, save to FMS
            logger.warning("Saving dataframes to FMS not yet implemented, saving locally instead.")
            stable_fixed_points_save_path = make_name_unique(
                dataframe_savedir / stable_fixed_points_file_name
            )
        stable_fixed_points_dataset.to_parquet(
            stable_fixed_points_save_path,
        )
        # add to DataframeManifest for stable fixed points for this dataset
        fixed_points_location = build_dataframe_location_from_path(stable_fixed_points_save_path)
        fixed_points_dataframe_manifest.locations[dataset_name] = fixed_points_location

    # save updated dataframe manifests for drift coefficients, grid points, and
    # stable fixed points with locations for each dataset
    for dataframe_manifest in [
        drift_dataframe_manifest,
        grid_dataframe_manifest,
        fixed_points_dataframe_manifest,
    ]:
        logger.info(
            "Saving dataframe manifest [ %s ] with updated locations for datasets [ %s ]",
            dataframe_manifest.name,
            list(dataframe_manifest.locations.keys()),
        )
        save_dataframe_manifest(dataframe_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
