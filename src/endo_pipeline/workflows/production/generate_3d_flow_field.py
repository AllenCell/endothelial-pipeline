from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    upload_to_fms: bool = False,
) -> None:
    """
    Generate 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for a given set of datasets.

    #dynamical-systems #diffae-feature-analysis

    **Workflow defaults**

    This workflow runs on features derived from the DiffAE model (specified by
    the default settings `DEFAULT_MODEL_MANIFEST_NAME` and
    `DEFAULT_MODEL_RUN_NAME`) as obtained from image crops of the specified
    `crop_pattern` type (i.e., grid-based or tracked-based crops). By default,
    it uses the time series of features extracted from grid-based crops but can
    also be run using features extracted from tracked-based crops by setting the
    `crop_pattern` parameter to "tracked".

    The specific features used for flow field estimation and analysis are
    determined by the `DYNAMICS_COLUMN_NAMES` setting, which specifies the names
    of the three features to use for flow field estimation and analysis. By
    default, these are set to be the polar angle, polar radius, and rho features
    derived from the DiffAE features via a 3D PCA transformation. For more
    details on the specific features used and how they are derived, see the
    methods `fit_pca` and `project_features_to_pcs` in the
    `pca` module.

    The workflow runs on the datasets specified via the `datasets` parameter,
    which can be a list of dataset names or dataset collection names. By
    default, it uses the datasets specified in the setting
    `DATASET_COLLECTION_FOR_3D_DYNAMICS`.

    **Flow field estimation and analysis**

    Using the 3D feature space defined by the DiffAE + PC derived features:

        (polar_theta, polar_r, rho)

    this workflow will do the following for each specified dataset:

    1. Estimate 3D flow fields using a kernel-based method for estimating
       Kramers-Moyal coefficients from time series data.
    2. Use interpolation to get a callable flow field function.
    3. Identify stable fixed points in the 3D flow field using a root-finding
       method applied to the flow field function.
    4. Save the following outputs for each dataset as parquet files:
        - Dataframe with the estimated drift coefficients at each grid point for
          each dataset.
        - Dataframe with the corresponding grid point coordinates for each
          dataset.
        - Dataframe with the stable fixed point locations for each dataset.

    Parameters
    ----------
    crop_pattern
        The crop pattern to use features from.
    datasets
        Optional, specific dataset(s) to run the workflow on.
    upload_to_fms
        If True, upload the output dataframes to FMS and update the
        corresponding dataframe manifests with the FMS locations. If False,
        save the output dataframes locally and log paths.
    """
    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_dataframe,
        make_name_unique,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        get_callable_vector_field,
        get_fixed_points_within_bounds,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
    from endo_pipeline.manifests import (
        DataframeLocation,
        build_dataframe_location_from_path,
        create_dataframe_manifest,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_THETA_RESCALED,
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        LOWER_PERCENTILE_FOR_STABLE_FP,
        NUM_INIT_SAMPLES,
        PAD_BINS_FLOAT,
        TIME_STEP_IN_MINUTES,
        UPPER_PERCENTILE_FOR_STABLE_FP,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_DRIFT,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        FMS_ANNOTATION_NOTES_DRIFT,
        FMS_ANNOTATION_NOTES_FIXED_POINTS,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names: list[ColumnName.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)
    drift_column_names: list[str] = [f"{name}_drift" for name in column_names]
    # columns to keep when loading dataframes
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    # Load default model manifest and get corresponding feature dataframe
    # manifest name for default run name and specified crop pattern.
    model_manifest = load_model_manifest(model_manifest_name)

    # Load dataframe manifest for the features to be used in flow field
    # estimation and analysis.
    base_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

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
    dataframe_savedir = get_output_path(__file__, crop_pattern)
    demo_suffix = "_demo" if DEMO_MODE else ""
    drift_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{base_name}{demo_suffix}"
    fixed_points_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}{demo_suffix}"
    )
    drift_dataframe_manifest = create_dataframe_manifest(
        drift_dataframe_manifest_name, workflow_name=__file__
    )
    fixed_points_dataframe_manifest = create_dataframe_manifest(
        fixed_points_dataframe_manifest_name, workflow_name=__file__
    )
    logger.info(
        "Dataframes with 3D flow field estimation results will be saved to: [ %s ]",
        dataframe_savedir,
    )

    # Default list of datasets if not provided. Filter by datasets available in
    # the manifest.
    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)
    if DEMO_MODE:
        logger.warning(
            "DEMO MODE: Processing no more than two of the provided datasets for quick testing."
        )
        # take min of the number of datasets provided and 2, to limit to at most
        # 2 datasets in DEMO_MODE for quick visualization (i.e., avoid error if
        # only 1 dataset is provided)
        num_datasets = min(len(dataset_names), 2)
        dataset_names = dataset_names[:num_datasets]

    # initialize list to hold dataframes of stable fixed points from all
    # datasets with columns for dataset name and 3D PC space coordinates
    stable_fixed_points_all_datasets_list = []

    # initialize kernels and bin widths for each of the three variables for flow
    # field estimation
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    rescaled_theta_period = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)

    # Get the corresponding kernels and bin widths for each variable. For the
    # polar angle variable, also specify the period for the kernel based on the
    # rescaled theta range, to ensure that the periodicity of the polar angle is
    # taken into account in the flow field estimation.
    for column_name in column_names:
        name = KERNEL_NAMES_DYNAMICS[column_name]
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS[column_name]
        period = rescaled_theta_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
        bin_width = BIN_WIDTHS_DYNAMICS[column_name]
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
        bin_widths.append(bin_width)

    # add parameters to dataframe manifests for traceability
    for output_dataframe_manifest in [
        drift_dataframe_manifest,
        fixed_points_dataframe_manifest,
    ]:
        output_dataframe_manifest.parameters = {
            "model_manifest_name": model_manifest_name,
            "run_name": run_name,
            "crop_pattern": crop_pattern,
            "columns": [column.value for column in column_names],
            "kernel_names": [kernel.name for kernel in kernels],
            "kernel_bandwidths": [kernel.bandwidth for kernel in kernels],
            "bin_widths": bin_widths,
            "num_init_samples_for_root_solver": NUM_INIT_SAMPLES,
            "lower_percentile_for_stable_fp": LOWER_PERCENTILE_FOR_STABLE_FP,
            "upper_percentile_for_stable_fp": UPPER_PERCENTILE_FOR_STABLE_FP,
        }
        save_dataframe_manifest(output_dataframe_manifest)

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found in manifest [ %s ] for dataset [ %s ]. Skipping this dataset.",
                feature_dataframe_manifest_name,
                dataset_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.shear_stress_regime) > 1:
            logger.warning(
                "Dataset [ %s ] has more than one shear stress condition: [ %s ]. "
                "Skipping for 3D flow field analysis.",
                dataset_name,
                dataset_config.shear_stress_regime,
            )
            continue

        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for flow field estimation and analysis to save memory.
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # get bins for flow field estimation based on the trajectories, to be
        # used for kernel-convolution-based estimation of the Kramers-Moyal
        # coefficients. The bins are determined by the specified bin widths and
        # the range of the data.
        bins, centers = get_bins(
            bin_widths,
            data=df_steady_state[column_names].to_numpy(),
            pad=PAD_BINS_FLOAT,
        )

        # get list of per-crop trajectories, the corresponding
        # displacement vectors, and time differences
        traj_list, d_traj_list = get_traj_and_diff(df_steady_state, column_names)

        # get drift estimates in units hours^-1 for each bin in 3D space
        # (Kramers-Moyal coefficient estimation)
        drift_coeffs = get_kramers_moyal_coeffs(
            traj_list, d_traj_list, bins=bins, dt=TIME_STEP_IN_MINUTES / 60, kernel=kernels
        )[0]
        feature_grid = np.meshgrid(*centers, indexing="ij")

        # build dataframe with columns for bin centers in each of the three
        # dimensions and the corresponding drift coefficients
        vector_field_df = pd.DataFrame(
            columns=[ColumnName.DATASET, *drift_column_names, *column_names]
        )
        for index, column_name, drift_column_name in zip(
            (0, 1, 2), column_names, drift_column_names, strict=True
        ):
            vector_field_df[column_name] = feature_grid[index].flatten()
            vector_field_df[drift_column_name] = drift_coeffs[..., index].flatten()
        vector_field_df[ColumnName.DATASET] = dataset_name

        # save drift coefficients and grid points dataframes to parquet files,
        # with names that include the input dataframe manifest name for
        # traceability and to avoid naming conflicts with other runs
        drift_coeffs_file_name = (
            f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{dataset_name}{demo_suffix}.parquet"
        )
        drift_coeffs_save_path = make_name_unique(dataframe_savedir / drift_coeffs_file_name)
        vector_field_df.to_parquet(drift_coeffs_save_path)
        logger.info(
            "Saved dataframe with drift coefficients and grid points locally to [ %s ]",
            drift_coeffs_save_path,
        )
        # Upload dataframes to FMS and update manifests
        if upload_to_fms:
            dataset_config = load_dataset_config(dataset_name)
            drift_annotations = build_fms_annotations(
                dataset_config,
                model_manifest=model_manifest,
                run_name=run_name,
                additional_notes=FMS_ANNOTATION_NOTES_DRIFT,
            )
            drift_fmsid = upload_file_to_fms(
                drift_coeffs_save_path, annotations=drift_annotations, file_type="parquet"
            )
            drift_dataframe_manifest.locations[dataset_name] = DataframeLocation(fmsid=drift_fmsid)
            logger.info(
                "Uploaded dataframe with drift coefficients and grid points for dataset [ %s ] to FMS with FMS ID [ %s ]",
                dataset_name,
                drift_fmsid,
            )
        # if not uploading to FMS, log only the path if there is no location for
        # that dataset or if there is, but the FMS ID is None
        elif (
            drift_dataframe_manifest.locations.get(dataset_name) is None
            or drift_dataframe_manifest.locations[dataset_name].fmsid is None
        ):
            drift_dataframe_manifest.locations[dataset_name] = build_dataframe_location_from_path(
                drift_coeffs_save_path
            )
        # save updated manifest with new locations (either FMS or local paths)
        save_dataframe_manifest(drift_dataframe_manifest)

        ## extrapolate the drift to get a flow field over the entire 3D space as specified by the input bins and centers
        extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
            drift_coeffs, centers, method="linear", for_vtk_files=False
        )

        # get callable drift function to be used for root finding to identify
        # fixed points
        drift_function = get_callable_vector_field(
            extrapolated_flow_field_dict_reg, for_solve_ivp=False, method="linear"
        )

        fixed_points_for_dataset = get_fixed_points_within_bounds(
            vector_field_function=drift_function,
            dataframe=df_steady_state,
            column_names=column_names,
            num_inits_for_root_solver=NUM_INIT_SAMPLES,
            lower_percentile=LOWER_PERCENTILE_FOR_STABLE_FP,
            upper_percentile=UPPER_PERCENTILE_FOR_STABLE_FP,
            polar_angle_range=BIN_LIMITS_THETA_RESCALED if RESCALE_THETA else (-np.pi, np.pi),
        )

        # add stable fixed points from this dataset to the overall dataframe
        # (checking first if returned dataframe is empty first to avoid issues
        # with concatenation and saving an empty dataframe)
        if fixed_points_for_dataset.empty:
            continue

        stable_fixed_points_all_datasets_list.append(fixed_points_for_dataset)

        # save stable fixed points from this dataset to parquet file
        fixed_points_file_name = (
            f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{dataset_name}{demo_suffix}.parquet"
        )
        fixed_points_save_path = make_name_unique(dataframe_savedir / fixed_points_file_name)
        fixed_points_for_dataset.to_parquet(fixed_points_save_path)
        logger.info("Saved dataframe of points locally to [ %s ]", fixed_points_save_path)
        # if uploading to FMS, update the dataframe manifest
        if upload_to_fms:
            fixed_points_annotations = build_fms_annotations(
                dataset_config,
                model_manifest=model_manifest,
                run_name=run_name,
                additional_notes=FMS_ANNOTATION_NOTES_FIXED_POINTS,
            )
            fixed_points_fmsid = upload_file_to_fms(
                fixed_points_save_path,
                annotations=fixed_points_annotations,
                file_type="parquet",
            )
            fixed_points_dataframe_manifest.locations[dataset_name] = DataframeLocation(
                fmsid=fixed_points_fmsid
            )
            logger.info(
                "Uploaded dataframe of stable fixed points for dataset [ %s ] to FMS with FMS ID [ %s ]",
                dataset_name,
                fixed_points_fmsid,
            )
        # else, log only the path if there is no location for that dataset or if
        # there is, but the FMS ID is None
        elif (
            fixed_points_dataframe_manifest.locations.get(dataset_name) is None
            or fixed_points_dataframe_manifest.locations[dataset_name].fmsid is None
        ):
            fixed_points_dataframe_manifest.locations[dataset_name] = DataframeLocation(
                path=fixed_points_save_path
            )
        # save updated manifest with new locations (either FMS or local paths)
        save_dataframe_manifest(fixed_points_dataframe_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
