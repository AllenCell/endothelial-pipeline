from endo_pipeline.cli import Datasets, PatchType, StrList


def main(
    patch_type: PatchType = "grid_based",
    columns: StrList | None = None,
    datasets: Datasets | None = None,
) -> None:
    """
    Generate drift vector field and estimate fixed points.

    #dynamical-systems #fixed-points #grid-based #cell-centered #test-ready

    This workflow generates the flow field based on the following features
    derived from evaluating the DiffAE model on specific patches of the data:

    - `polar_theta` = polar angle coordinate computed from PC1 and PC2
    - `polar_r` = polar radius coordinate computed from PC1 and PC2
    - `rho` = PC3 value with sign flipped

    Any combination of these features can be used to generate the flow field
    with the corresponding dimensionality. By default, the workflow will
    generate the a 3D flow field using all three features.

    Using the feature space defined by the specified input features, this
    workflow does the following for each specified dataset:

    1. Estimate drift flow fields using a kernel-based method for estimating
       Kramers-Moyal coefficients from time series data
    2. Use interpolation to get a callable flow field function
    3. Identify stable fixed points of the flow field using a root-finding
       method applied to the flow field function
    4. Save the following outputs for each dataset as parquet files:
        - Dataframe with the drift vector field
        - Dataframe with the drift fixed points

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe generate-flow-field -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe generate-flow-field --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will generate the
    flow field for the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    columns
        Specific columns to use to generate flow field.
    datasets
        List of datasets or dataset collections to generate flow fields for.
    """

    import logging

    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_drift_estimates_and_fixed_points,
        get_valid_flow_field_column_names,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        KERNEL_PERIODS_DYNAMICS,
        LOWER_PERCENTILE_FOR_FILTERING_FPTS,
        METADATA_COLUMNS_TO_KEEP,
        NUM_INIT_SAMPLES,
        TIME_STEP_IN_HOURS,
        UPPER_PERCENTILE_FOR_FILTERING_FPTS,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        FMS_ANNOTATION_NOTES_FIXED_POINTS,
        FMS_ANNOTATION_NOTES_VECTOR_FIELD,
    )
    from endo_pipeline.settings.manifest_names import (
        DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD,
        FIXED_POINT_MANIFEST_NAMES,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Workflow only supports generating flow fields from combinations of
    # three specific column names (as defined in DYNAMICS_COLUMN_NAMES). If
    # column names are provided, ensure they are a subset of these columns and
    # skip any that are not. If no column names are provided, just use these
    # three columns.
    column_names = get_valid_flow_field_column_names(columns)

    if not column_names:
        logger.error("No valid columns for generating flow field.")
        return

    logger.info("Generating flow field for columns: %s", column_names)

    # Columns to keep when loading feature dataframe
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load default model manifest and corresponding feature dataframe for
    # specified patch type
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Build dataframe manifest names that include sorted list of selected
    # columns used to generate the flow field.
    name_suffix = join_sorted_strings(column_names)
    vector_field_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD}_{name_suffix}_{patch_type}"
    )
    fixed_points_dataframe_manifest_name = f"{FIXED_POINT_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    vector_field_dataframe_manifest = create_dataframe_manifest(
        vector_field_dataframe_manifest_name, workflow_name=__file__
    )
    fixed_points_dataframe_manifest = create_dataframe_manifest(
        fixed_points_dataframe_manifest_name, workflow_name=__file__
    )

    # Initialize kernels and bin widths for each selected column
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    for column_name in column_names:
        kernels.append(
            KramersMoyalKernel(
                name=KERNEL_NAMES_DYNAMICS[column_name],
                bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                period=KERNEL_PERIODS_DYNAMICS[column_name],
            )
        )
        bin_widths.append(BIN_WIDTHS_DYNAMICS[column_name])

    # Add parameters to dataframe manifests for traceability
    for output_dataframe_manifest in [
        vector_field_dataframe_manifest,
        fixed_points_dataframe_manifest,
    ]:
        output_dataframe_manifest.parameters = {
            "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
            "run_name": DEFAULT_MODEL_RUN_NAME,
            "patch_type": patch_type,
            "kernels": [
                {
                    "column": str(column),
                    "name": kernel.name,
                    "bandwidth": kernel.bandwidth,
                    "period": kernel.period,
                }
                for column, kernel in zip(column_names, kernels, strict=False)
            ],
            "bin_widths": bin_widths,
            "num_init_samples_for_root_solver": NUM_INIT_SAMPLES,
            "lower_percentile_for_filtering_fpts": LOWER_PERCENTILE_FOR_FILTERING_FPTS,
            "upper_percentile_for_filtering_fpts": UPPER_PERCENTILE_FOR_FILTERING_FPTS,
        }
        save_dataframe_manifest(output_dataframe_manifest)

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                feature_dataframe_manifest_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)

        # Load feature dataframe for dataset with only the required columns and
        # filter out non-steady-state timepoints
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # Generate vector field and calculate fixed points per flow condition
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

            # Append vector field dataframe to list to be concatenated outside
            # of flow condition loop
            vector_field_dataframe_list.append(vector_field_dataframe)

            # Append fixed points dataframe to list to be concatenated outside
            # of flow condition loop, if the dataframe is not empty
            if not fixed_points_dataframe.empty:
                fixed_points_dataframe_list.append(fixed_points_dataframe)

        # Concatenate vector fields for flow conditions into single dataframe.
        vector_field_for_dataset = pd.concat(vector_field_dataframe_list, ignore_index=True)

        # If there are any fixed points for the dataset, concatenate them into
        # a single dataframe.
        if fixed_points_dataframe_list:
            fixed_points_for_dataset = pd.concat(fixed_points_dataframe_list, ignore_index=True)
        else:
            fixed_points_for_dataset = None

        for manifest, dataframe, name_prefix, additional_notes in [
            (
                vector_field_dataframe_manifest,
                vector_field_for_dataset,
                DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD,
                FMS_ANNOTATION_NOTES_VECTOR_FIELD % len(column_names),
            ),
            (
                fixed_points_dataframe_manifest,
                fixed_points_for_dataset,
                FIXED_POINT_MANIFEST_NAMES[patch_type],
                FMS_ANNOTATION_NOTES_FIXED_POINTS % len(column_names),
            ),
        ]:
            # Skip if dataframe is None.
            if dataframe is None:
                continue

            # Save dataframe to file
            save_path = output_path / f"{name_prefix}_{dataset_name}{name_suffix}.parquet"
            dataframe.to_parquet(save_path, index=False)

            # Create location object with output path
            location = manifest.locations.get(dataset_name, DataframeLocation())
            location.path = save_path

            # Upload to FMS (internal only) and replace local path with file id
            if UPLOAD_TO_FMS:
                annotations = build_fms_annotations(
                    dataset_config,
                    model_manifest=model_manifest,
                    run_name=DEFAULT_MODEL_RUN_NAME,
                    additional_notes=additional_notes,
                )
                fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
                location.fmsid = fmsid
                location.path = None

            # Add dataframe location to dataframe manifest and save
            manifest.locations[dataset_name] = location
            save_dataframe_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
