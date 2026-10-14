from endo_pipeline.cli import Datasets, PatchType, StrList


def main(
    patch_type: PatchType = "grid_based",
    columns: StrList | None = None,
    datasets: Datasets | None = None,
) -> None:
    """
    Compute the state-dependent noise amplitude tensor sigma(x) from a
    histogram-based average of the residual autocovariance.

    This workflow estimates `sigma(x)` directly from
    `<eta(t) eta(t)^T | x(t) = x>`, a simple histogram average (no kernel
    smoothing) of the outer product of the drift-removed residuals `eta(t)`
    (the forward difference minus the drift predicted by the pre-computed
    vector field from `generate-flow-field`), binned by the state `x(t)`.
    This is an independent, non-parametric check of the kernel-convolution-based
    Kramers-Moyal diffusion tensor estimate, computed on the same grid as the
    pre-computed vector field.

    The resulting `sigma(x)` can be combined with the pre-computed Ito drift to
    assemble the Stratonovich (alpha=1/2) noise-induced drift correction (see
    `generate-stratonovich-flow-field`), so that the Ito and Stratonovich fixed
    points can be compared directly.

    #dynamical-systems #correlation-analysis #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-noise-amplitude -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-noise-amplitude --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    noise amplitude analysis for the first dataset, using only a subset of the
    patches available in the dataset.

    Parameters
    ----------
    patch_type
        Patch type used to compute the noise amplitude tensor.
    columns
        Specific columns to use to compute the noise amplitude tensor. Must
        match the columns used for a pre-computed vector field (see
        `generate-flow-field`).
    datasets
        List of datasets or dataset collections to compute the noise
        amplitude tensor for.
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
        filter_dataframe_by_track_length,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_track_duration_to_dataframe,
    )
    from endo_pipeline.library.analyze.numerics.correlations import (
        NoiseCorrelationCentering,
        compute_centered_residuals,
        get_trajectories_and_differences_for_noise_correlations,
    )
    from endo_pipeline.library.analyze.sde_interpretation import (
        compute_binned_residual_covariance,
        compute_histogram_noise_amplitude,
        create_noise_amplitude_vector_field_df,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_valid_flow_field_column_names,
        get_vector_field_as_dict_from_dataframe,
    )
    from endo_pipeline.library.analyze.vector_field_function import get_callable_vector_field
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
        MIN_SAMPLES_PER_NOISE_AMPLITUDE_BIN,
        TIME_STEP_IN_HOURS,
    )
    from endo_pipeline.settings.flow_field_dataframes import FMS_ANNOTATION_NOTES_NOISE_AMPLITUDE
    from endo_pipeline.settings.manifest_names import (
        NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES,
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

    max_num_patches = None
    if DEMO_MODE:
        max_num_patches = 50
        dataset_names = dataset_names[:1]
        logger.warning(
            "DEMO MODE - Limiting to %d dataset(s) and %d patches per dataset.",
            len(dataset_names),
            max_num_patches,
        )

    # Restrict to the columns supported for flow field estimation, since the
    # noise amplitude tensor is computed on the grid of a pre-computed vector
    # field over these same columns.
    column_names = get_valid_flow_field_column_names(columns)

    if not column_names:
        logger.error("No valid columns for computing noise amplitude.")
        return

    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]
    mesh_column_names = [ColumnTemplate.MESH_GRID % name for name in column_names]
    bin_widths = [BIN_WIDTHS_DYNAMICS[column_name] for column_name in column_names]

    # Load feature dataframe for specified patch type
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load pre-computed vector field dataframe manifest
    name_suffix = join_sorted_strings(column_names)
    vector_field_manifest_name = f"{VECTOR_FIELD_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)

    # Build dataframe manifest names that include sorted list of selected
    # columns used to compute the noise amplitude tensor.
    noise_amplitude_name_prefix = NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES[patch_type]
    noise_amplitude_manifest_name = f"{noise_amplitude_name_prefix}_{name_suffix}"
    noise_amplitude_manifest = create_dataframe_manifest(
        noise_amplitude_manifest_name, workflow_name=__file__
    )

    # Add parameters to dataframe manifest for traceability
    noise_amplitude_manifest.parameters = {
        "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
        "run_name": DEFAULT_MODEL_RUN_NAME,
        "patch_type": patch_type,
        "columns": [f"{column}" for column in column_names],
        "min_samples_per_bin": MIN_SAMPLES_PER_NOISE_AMPLITUDE_BIN,
    }
    save_dataframe_manifest(noise_amplitude_manifest)

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

        # load dataframe and filter to just steady state timepoints
        df_delayed = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_delayed[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # load pre-computed vector field dataframe for the current dataset
        df_vec = load_dataframe(vector_field_manifest.locations[dataset_name], delay=False)

        # process on a per-flow condition basis
        noise_amplitude_dataframe_list = []
        for flow_condition in dataset_config.flow_conditions:
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"

            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            steady_state_duration = (
                df_flow[Column.TIMEPOINT].max() - df_flow[Column.TIMEPOINT].min()
            )
            # Filter tracks by duration
            track_duration_filter = min(LONG_TRACK_THRESHOLD_LENGTH, steady_state_duration)
            df_flow = add_track_duration_to_dataframe(
                df_flow, grouping_columns=[Column.CROP_INDEX], time_column=Column.TIMEPOINT
            )
            df_flow = filter_dataframe_by_track_length(
                dataframe=df_flow, minimum_track_length=track_duration_filter
            )

            # Get pre-computed vector field for the current flow condition
            vector_field_for_flow_condition = filter_dataframe_by_shear_stress(
                df_vec, flow_condition.shear_stress
            )
            if vector_field_for_flow_condition.empty:
                logger.warning(
                    "No pre-computed vector field found for [ %s ]. Skipping.",
                    dataset_name_flow,
                )
                continue

            vector_field_dict = get_vector_field_as_dict_from_dataframe(
                vector_field_for_flow_condition, column_names
            )
            vector_field = get_callable_vector_field(vector_field_dict, for_solve_ivp=False)

            # Reconstruct the bin edges of the pre-computed vector field grid
            # (stored as bin centers), so that the noise amplitude tensor is
            # computed on the exact same grid and can later be combined with
            # the drift vector field (e.g. for the noise-induced drift term).
            grid_points_1d = [
                np.sort(vector_field_for_flow_condition[column_name].unique())
                for column_name in mesh_column_names
            ]
            bin_limits = [
                (
                    grid_points_1d[i][0] - bin_widths[i] / 2,
                    grid_points_1d[i][-1] + bin_widths[i] / 2,
                )
                for i in range(len(column_names))
            ]
            num_bins = [len(points) for points in grid_points_1d]
            bin_edges = [
                np.linspace(bin_limit[0], bin_limit[1], num_bin + 1)
                for bin_limit, num_bin in zip(bin_limits, num_bins, strict=True)
            ]

            timepoints_array = np.asarray(sorted(df_flow[Column.TIMEPOINT].unique()))

            # Get the per-patch trajectories and their forward differences
            traj_list, d_traj_list = get_trajectories_and_differences_for_noise_correlations(
                df_flow, column_names, max_num_patches=max_num_patches
            )
            if not traj_list or not d_traj_list:
                logger.warning("No valid patches found for [ %s ]. Skipping.", dataset_name_flow)
                continue

            # Get the drift-removed residuals eta(t), along with the state
            # x(t) each residual was observed at
            residuals, positions = compute_centered_residuals(
                traj_list=traj_list,
                d_traj_list=d_traj_list,
                column_names=column_names,
                timepoints_array=timepoints_array,
                vector_field=vector_field,
                centering_method=NoiseCorrelationCentering.DRIFT_SUBTRACTED,
                return_positions=True,
            )

            # Histogram-based average of <eta(t) eta(t)^T | x(t) = x>, and the
            # corresponding noise amplitude tensor sigma(x)
            covariance, sample_counts = compute_binned_residual_covariance(
                positions=positions, residuals=residuals, bins=bin_edges
            )
            noise_amplitude = compute_histogram_noise_amplitude(
                covariance=covariance,
                counts=sample_counts,
                time_step=TIME_STEP_IN_HOURS,
                min_samples_per_bin=MIN_SAMPLES_PER_NOISE_AMPLITUDE_BIN,
            )

            feature_grid = np.meshgrid(*grid_points_1d, indexing="ij")
            metadata_dict = {
                Column.DATASET: dataset_name,
                Column.SHEAR_STRESS: flow_condition.shear_stress,
            }
            noise_amplitude_dataframe_list.append(
                create_noise_amplitude_vector_field_df(
                    noise_amplitude=noise_amplitude,
                    sample_counts=sample_counts,
                    column_names=column_names,
                    feature_grid=feature_grid,
                    metadata_dict=metadata_dict,
                )
            )

        if not noise_amplitude_dataframe_list:
            logger.warning("No noise amplitude tensor computed for [ %s ]. Skipping.", dataset_name)
            continue

        # Concatenate noise amplitude dataframes for each flow condition
        noise_amplitude_for_dataset = pd.concat(noise_amplitude_dataframe_list, ignore_index=True)

        # Save dataframe to file
        save_path = (
            output_path / f"{noise_amplitude_name_prefix}_{dataset_name}_{name_suffix}.parquet"
        )
        noise_amplitude_for_dataset.to_parquet(save_path, index=False)

        # Create location object with output path
        location = noise_amplitude_manifest.locations.get(dataset_name, DataframeLocation())
        location.path = save_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(
                dataset_config,
                model_manifest=load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME),
                run_name=DEFAULT_MODEL_RUN_NAME,
                additional_notes=FMS_ANNOTATION_NOTES_NOISE_AMPLITUDE % len(column_names),
            )
            fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None

        # Add dataframe location to dataframe manifest and save
        noise_amplitude_manifest.locations[dataset_name] = location
        save_dataframe_manifest(noise_amplitude_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
