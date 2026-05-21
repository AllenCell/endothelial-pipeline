from endo_pipeline.cli import CropPattern, Datasets, StrList


def main(
    crop_pattern: CropPattern = "grid",
    columns: StrList | None = None,
    datasets: Datasets | None = None,
    upload_to_fms: bool = False,
) -> None:
    """
    Generate drift vector fields for the dynamics of the crop-based DiffAE
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
    determined by the input `columns`. By default, these are set to be the polar
    angle, polar radius, and rho features derived from the DiffAE features via a
    3D PCA transformation. For more details on the specific features used and
    how they are derived, see the methods `fit_pca` and
    `project_features_to_pcs` in the `pca` module.

    Note that the number of input features determines the dimensionality of the
    flow field and fixed point analysis. By default, the workflow is set to use
    three features for a 3D flow field, which can then be visualized using the
    `visualize_3d_flow_field` workflow. However, the workflow can also be run
    using only a subset of these features (e.g., just the polar angle and polar
    radius) for a 2D flow field analysis, which can be visualized using the
    `visualize_2d_flow_field` workflow. If using a subset of the default three
    features, make sure to specify the corresponding columns in the `columns`
    parameter and to use the appropriate visualization workflow for the number
    of features used.

    The workflow runs on the datasets specified via the `datasets` parameter,
    which can be a list of dataset names or dataset collection names. By
    default, it uses the datasets specified in the setting
    `DATASET_COLLECTION_FOR_3D_DYNAMICS`.

    **Flow field estimation and analysis**

    Using the feature space defined by the specified input features, this
    workflow does the following for each specified dataset:

    1. Estimate drift flow fields using a kernel-based method for estimating
       Kramers-Moyal coefficients from time series data.
    2. Use interpolation to get a callable flow field function.
    3. Identify stable fixed points of the flow field using a root-finding
       method applied to the flow field function.
    4. Save the following outputs for each dataset as parquet files:
        - Dataframe with the estimated drift coefficients at each grid point for
          each dataset.
        - Dataframe with the corresponding grid point coordinates for each
          dataset.
        - Dataframe with the stable fixed point locations for each dataset.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will generate the
    flow field for the first dataset.

    Parameters
    ----------
    crop_pattern
        The crop pattern to use features from.
    datasets
        Optional, specific dataset(s) to run the workflow on.
    columns
        Optional, specific columns (features) to use for flow field estimation
        and analysis.
    upload_to_fms
        If True, upload the output dataframes to FMS and update the
        corresponding dataframe manifests with the FMS locations. If False, save
        the output dataframes locally and log paths.
    """
    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        make_name_unique,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_drift_estimates_and_fixed_points,
    )
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
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        LOWER_PERCENTILE_FOR_FILTERING_FPTS,
        METADATA_COLUMNS_TO_KEEP,
        NUM_INIT_SAMPLES,
        POLAR_ANGLE_PERIOD,
        RESCALE_THETA,
        TIME_STEP_IN_HOURS,
        UPPER_PERCENTILE_FOR_FILTERING_FPTS,
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
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Workflow only supports generating flow fields from combinations of
    # three specific column names (as defined in DYNAMICS_COLUMN_NAMES). If
    # column names are provided, ensure they are a subset of these columns and
    # skip any that are not. If no column names are provided, just use these
    # three columns.
    column_names = []
    if columns is not None:
        for column in columns:
            if column in DYNAMICS_COLUMN_NAMES:
                column_names.append(ColumnName.DiffAEData(column))
            else:
                logger.warning("Column '%s' not supported for flow fields. Skipping.", column)
    else:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    logger.info("Generating flow field for columns: %s", column_names)

    # Columns to keep when loading dataframes
    ndim = len(column_names)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    # Load default model manifest and corresponding feature dataframe for
    # specified crop pattern.
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[crop_pattern]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Also build dataframe manifests for the outputs of this workflow (drift
    # coefficients and identified fixed points) with names that include the
    # input dataframe manifest name for traceability and to avoid naming
    # conflicts with other runs.
    dataframe_savedir = get_output_path(__file__, crop_pattern)

    # Build dataframe manifest names that include sorted list of selected
    # columns used to generate the flow field.
    name_suffix = "_demo" if DEMO_MODE else ""
    name_suffix = f"_{join_sorted_strings(column_names)}_{crop_pattern}{name_suffix}"
    drift_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}{name_suffix}"
    fixed_points_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix}"
    drift_dataframe_manifest = create_dataframe_manifest(
        drift_dataframe_manifest_name, workflow_name=__file__
    )
    fixed_points_dataframe_manifest = create_dataframe_manifest(
        fixed_points_dataframe_manifest_name, workflow_name=__file__
    )

    # initialize kernels and bin widths for each of the three variables for flow
    # field estimation
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    rescaled_theta_period = POLAR_ANGLE_PERIOD + np.pi * (1 - RESCALE_THETA)

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
    column_names_yaml_safe = [f"{column}" for column in column_names]
    for output_dataframe_manifest in [
        drift_dataframe_manifest,
        fixed_points_dataframe_manifest,
    ]:
        output_dataframe_manifest.parameters = {
            "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
            "run_name": DEFAULT_MODEL_RUN_NAME,
            "crop_pattern": crop_pattern,
            "columns": column_names_yaml_safe,
            "kernel_names": [kernel.name for kernel in kernels],
            "kernel_bandwidths": [kernel.bandwidth for kernel in kernels],
            "bin_widths": bin_widths,
            "num_init_samples_for_root_solver": NUM_INIT_SAMPLES,
            "lower_percentile_for_filtering_fpts": LOWER_PERCENTILE_FOR_FILTERING_FPTS,
            "upper_percentile_for_filtering_fpts": UPPER_PERCENTILE_FOR_FILTERING_FPTS,
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

        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for flow field estimation and analysis to save memory.
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # process on a per-flow condition basis
        vector_field_dataframe_list = []
        fixed_points_dataframe_list = []
        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            metadata_dict = {
                ColumnName.DATASET: dataset_name,
                ColumnName.SHEAR_STRESS: shear_stress,
            }
            vector_field_dataframe, fixed_points_dataframe = get_drift_estimates_and_fixed_points(
                dataframe=df_flow,
                column_names=column_names,
                bin_widths=bin_widths,
                kernel=kernels,
                time_step=TIME_STEP_IN_HOURS,
                metadata_dict=metadata_dict,
            )

            # Append to lists to be concatenated outside of flow condition loop
            vector_field_dataframe_list.append(vector_field_dataframe)

            # (checking first if returned fixed point dataframe dataframe is
            # empty to avoid issues with concatenation and saving an empty
            # dataframe)
            if fixed_points_dataframe.empty:
                continue
            fixed_points_dataframe_list.append(fixed_points_dataframe)

        # save drift coefficients and grid points dataframes to parquet files,
        # with names that include the input dataframe manifest name for
        # traceability and to avoid naming conflicts with other runs
        vector_field_for_dataset = pd.concat(vector_field_dataframe_list, ignore_index=True)
        vector_field_file_name = (
            f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{dataset_name}{name_suffix}.parquet"
        )
        vector_field_save_path = make_name_unique(dataframe_savedir / vector_field_file_name)
        vector_field_for_dataset.to_parquet(vector_field_save_path)
        # Upload dataframes to FMS and update manifests
        if upload_to_fms:
            dataset_config = load_dataset_config(dataset_name)
            drift_annotations = build_fms_annotations(
                dataset_config,
                model_manifest=model_manifest,
                run_name=DEFAULT_MODEL_RUN_NAME,
                additional_notes=FMS_ANNOTATION_NOTES_DRIFT,
            )
            vector_field_fmsid = upload_file_to_fms(
                vector_field_save_path, annotations=drift_annotations, file_type="parquet"
            )
            drift_dataframe_manifest.locations[dataset_name] = DataframeLocation(
                fmsid=vector_field_fmsid
            )
        # if not uploading to FMS, log only the path if there is no location for
        # that dataset or if there is, but the FMS ID is None
        elif (
            drift_dataframe_manifest.locations.get(dataset_name) is None
            or drift_dataframe_manifest.locations[dataset_name].fmsid is None
        ):
            drift_dataframe_manifest.locations[dataset_name] = build_dataframe_location_from_path(
                vector_field_save_path
            )
        # save updated manifest with new locations (either FMS or local paths)
        save_dataframe_manifest(drift_dataframe_manifest)

        # if there are fixed points for this dataset, concatenate them into a
        # single dataframe and add to list of dataframes of fixed points for all
        # datasets
        if not fixed_points_dataframe_list:
            continue

        fixed_points_for_dataset = pd.concat(fixed_points_dataframe_list, ignore_index=True)

        # save stable fixed points from this dataset to parquet file
        fixed_points_file_name = (
            f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{dataset_name}{name_suffix}.parquet"
        )
        fixed_points_save_path = make_name_unique(dataframe_savedir / fixed_points_file_name)
        fixed_points_for_dataset.to_parquet(fixed_points_save_path)
        # if uploading to FMS, update the dataframe manifest
        if upload_to_fms:
            fixed_points_annotations = build_fms_annotations(
                dataset_config,
                model_manifest=model_manifest,
                run_name=DEFAULT_MODEL_RUN_NAME,
                additional_notes=FMS_ANNOTATION_NOTES_FIXED_POINTS[ndim],
            )
            fixed_points_fmsid = upload_file_to_fms(
                fixed_points_save_path,
                annotations=fixed_points_annotations,
                file_type="parquet",
            )
            fixed_points_dataframe_manifest.locations[dataset_name] = DataframeLocation(
                fmsid=fixed_points_fmsid
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
