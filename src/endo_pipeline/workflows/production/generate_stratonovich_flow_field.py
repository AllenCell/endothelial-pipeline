from endo_pipeline.cli import Datasets, PatchType, StrList


def main(
    patch_type: PatchType = "grid_based",
    columns: StrList | None = None,
    datasets: Datasets | None = None,
) -> None:
    """
    Generate the Stratonovich-corrected drift vector field and estimate its fixed points.

    This workflow loads a pre-computed Ito drift vector field (see
    `generate-flow-field`) and a pre-computed noise amplitude tensor field
    `sigma(x)` (see `compute-noise-amplitude`), both defined on the same grid,
    and assembles the Stratonovich (alpha=1/2) noise-induced drift correction:

        f_i^(1/2) = f_i^(Ito) - (1/2) * sum_{j,k} sigma_kj * d(sigma_ij)/d(x_k)

    It then behaves like `generate-flow-field` in that it:

    1. Uses interpolation to get a callable corrected flow field function
    2. Identifies stable fixed points of the corrected flow field using a
       root-finding method applied to the flow field function
    3. Saves the following outputs for each dataset as parquet files:
        - Dataframe with the Stratonovich-corrected drift vector field
        - Dataframe with the Stratonovich-corrected drift fixed points

    #dynamical-systems #fixed-points #correlation-analysis #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe generate-stratonovich-flow-field -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe generate-stratonovich-flow-field --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will generate the
    Stratonovich-corrected flow field for the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    columns
        Specific columns to use. Must match the columns used for the
        pre-computed Ito vector field (see `generate-flow-field`) and noise
        amplitude tensor (see `compute-noise-amplitude`).
    datasets
        List of datasets or dataset collections to generate corrected flow
        fields for.
    """
    import logging

    import numpy as np
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
        filter_dataframe_by_shear_stress,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.numerics.fixed_points import get_fixed_points_within_bounds
    from endo_pipeline.library.analyze.sde_interpretation import (
        compute_noise_induced_drift_residual,
        extrapolate_noise_amplitude_tensor,
        get_reshaped_noise_amplitude_and_grid,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        create_drift_vector_field_df,
        get_reshaped_vector_field_and_grid,
        get_valid_flow_field_column_names,
    )
    from endo_pipeline.library.analyze.vector_field_function import (
        compute_extrapolated_vector_field,
        get_callable_vector_field,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        LOWER_PERCENTILE_FOR_FILTERING_FPTS,
        METADATA_COLUMNS_TO_KEEP,
        NUM_INIT_SAMPLES,
        UPPER_PERCENTILE_FOR_FILTERING_FPTS,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        FMS_ANNOTATION_NOTES_STRATONOVICH_FIXED_POINTS,
        FMS_ANNOTATION_NOTES_STRATONOVICH_VECTOR_FIELD,
    )
    from endo_pipeline.settings.manifest_names import (
        NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES,
        STRATONOVICH_FIXED_POINT_MANIFEST_NAMES,
        STRATONOVICH_VECTOR_FIELD_MANIFEST_NAMES,
        VECTOR_FIELD_MANIFEST_NAMES,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or [
        *get_datasets_in_collection("shear_stress"),
        *get_datasets_in_collection("perturbation"),
    ]

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    column_names = get_valid_flow_field_column_names(columns)

    if not column_names:
        logger.error("No valid columns for generating Stratonovich flow field.")
        return

    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load default model manifest and corresponding feature dataframe for
    # specified patch type (needed for fixed point density sampling)
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load pre-computed Ito vector field and noise amplitude tensor manifests
    name_suffix = join_sorted_strings(column_names)
    vector_field_manifest_name = f"{VECTOR_FIELD_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)
    noise_amplitude_manifest_name = (
        f"{NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES[patch_type]}_{name_suffix}"
    )
    noise_amplitude_manifest = load_dataframe_manifest(noise_amplitude_manifest_name)

    # Build dataframe manifest names that include sorted list of selected
    # columns used to generate the corrected flow field.
    stratonovich_vector_field_manifest_name = (
        f"{STRATONOVICH_VECTOR_FIELD_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    )
    stratonovich_fixed_points_manifest_name = (
        f"{STRATONOVICH_FIXED_POINT_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    )
    stratonovich_vector_field_manifest = create_dataframe_manifest(
        stratonovich_vector_field_manifest_name, workflow_name=__file__
    )
    stratonovich_fixed_points_manifest = create_dataframe_manifest(
        stratonovich_fixed_points_manifest_name, workflow_name=__file__
    )

    # Add parameters to dataframe manifests for traceability
    for output_dataframe_manifest in [
        stratonovich_vector_field_manifest,
        stratonovich_fixed_points_manifest,
    ]:
        output_dataframe_manifest.parameters = {
            "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
            "run_name": DEFAULT_MODEL_RUN_NAME,
            "patch_type": patch_type,
            "columns": [f"{column}" for column in column_names],
            "alpha": 0.5,
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

        if dataset_name not in vector_field_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in vector field manifest '%s'. Skipping.",
                dataset_name,
                vector_field_manifest_name,
            )
            continue

        if dataset_name not in noise_amplitude_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in noise amplitude manifest '%s'. Skipping.",
                dataset_name,
                noise_amplitude_manifest_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)

        # Load feature dataframe for dataset, used for density-based sampling
        # of root solver initial conditions and percentile-based fixed point
        # filtering, filtered to just steady state timepoints
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # Load pre-computed Ito vector field and noise amplitude tensor for dataset
        df_vec = load_dataframe(vector_field_manifest.locations[dataset_name], delay=False)
        df_noise_amplitude = load_dataframe(
            noise_amplitude_manifest.locations[dataset_name], delay=False
        )

        # Generate corrected vector field and calculate fixed points per flow condition
        vector_field_dataframe_list = []
        fixed_points_dataframe_list = []

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"

            vector_field_for_flow_condition = filter_dataframe_by_shear_stress(df_vec, shear_stress)
            noise_amplitude_for_flow_condition = filter_dataframe_by_shear_stress(
                df_noise_amplitude, shear_stress
            )
            if vector_field_for_flow_condition.empty or noise_amplitude_for_flow_condition.empty:
                logger.warning(
                    "No pre-computed vector field or noise amplitude tensor found for "
                    "[ %s ]. Skipping.",
                    dataset_name_flow,
                )
                continue

            # Reconstruct the Ito drift and noise amplitude tensor fields on
            # their shared grid, and assemble the Stratonovich correction
            drift_coeffs, grid_points_1d = get_reshaped_vector_field_and_grid(
                vector_field_for_flow_condition, column_names
            )
            noise_amplitude, _ = get_reshaped_noise_amplitude_and_grid(
                noise_amplitude_for_flow_condition, column_names
            )
            # Fill NaN bins before differentiating, so sparse per-bin estimates
            # don't cascade into an almost entirely NaN corrected field
            noise_amplitude_filled = extrapolate_noise_amplitude_tensor(
                noise_amplitude, grid_points_1d
            )
            noise_induced_drift = compute_noise_induced_drift_residual(
                noise_amplitude_filled, grid_points_1d
            )
            stratonovich_drift_coeffs = drift_coeffs - 0.5 * noise_induced_drift

            metadata_dict = {
                Column.DATASET: dataset_name,
                Column.SHEAR_STRESS: shear_stress,
            }
            feature_grid = np.meshgrid(*grid_points_1d, indexing="ij")
            vector_field_dataframe = create_drift_vector_field_df(
                drift_coeffs=stratonovich_drift_coeffs,
                column_names=column_names,
                feature_grid=feature_grid,
                metadata_dict=metadata_dict,
            )
            vector_field_dataframe_list.append(vector_field_dataframe)

            # Use interpolation to get a callable corrected flow field function,
            # and identify its stable fixed points via root-finding
            extrapolated_flow_field_dict = compute_extrapolated_vector_field(
                stratonovich_drift_coeffs, grid_points_1d, method="linear", for_vtk_files=False
            )
            drift_function = get_callable_vector_field(
                extrapolated_flow_field_dict, for_solve_ivp=False, method="linear"
            )
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            fixed_points_dataframe = get_fixed_points_within_bounds(
                vector_field_function=drift_function,
                dataframe=df_flow,
                column_names=column_names,
                metadata_dict=metadata_dict,
            )

            # Append fixed points dataframe to list to be concatenated outside
            # of flow condition loop, if the dataframe is not empty
            if not fixed_points_dataframe.empty:
                fixed_points_dataframe_list.append(fixed_points_dataframe)

        if not vector_field_dataframe_list:
            logger.warning(
                "No Stratonovich vector field computed for [ %s ]. Skipping.", dataset_name
            )
            continue

        # Concatenate corrected vector fields for flow conditions into single dataframe.
        vector_field_for_dataset = pd.concat(vector_field_dataframe_list, ignore_index=True)

        # If there are any fixed points for the dataset, concatenate them into
        # a single dataframe.
        if fixed_points_dataframe_list:
            fixed_points_for_dataset = pd.concat(fixed_points_dataframe_list, ignore_index=True)
        else:
            fixed_points_for_dataset = None

        for manifest, dataframe, name_prefix, additional_notes in [
            (
                stratonovich_vector_field_manifest,
                vector_field_for_dataset,
                STRATONOVICH_VECTOR_FIELD_MANIFEST_NAMES[patch_type],
                FMS_ANNOTATION_NOTES_STRATONOVICH_VECTOR_FIELD % len(column_names),
            ),
            (
                stratonovich_fixed_points_manifest,
                fixed_points_for_dataset,
                STRATONOVICH_FIXED_POINT_MANIFEST_NAMES[patch_type],
                FMS_ANNOTATION_NOTES_STRATONOVICH_FIXED_POINTS % len(column_names),
            ),
        ]:
            # Skip if dataframe is None.
            if dataframe is None:
                continue

            # Save dataframe to file
            save_path = output_path / f"{name_prefix}_{dataset_name}_{name_suffix}.parquet"
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
