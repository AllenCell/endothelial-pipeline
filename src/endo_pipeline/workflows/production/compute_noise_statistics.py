from endo_pipeline.cli import Datasets, PatchType


def main(patch_type: PatchType = "grid_based", datasets: Datasets | None = None) -> None:
    """
    Compute noise statistics from feature displacements at steady state.

    This workflow computes the following from the centered residuals of feature
    displacements at steady state:
        - Cross-correlations of the noise term across time points as a function
          of time lag
        - Measurement noise covariance for each feature (obtained from the lag-1
          covariance)

    In addition to the tidy noise correlation / measurement noise covariance
    statistics dataframe, this workflow also saves the full two-timepoint noise
    correlation functions (see `compute-noise-correlations`) and the periodogram
    of the drift-removed residuals for each feature. These are loaded directly by
    `visualize-noise-statistics` to produce the corresponding plots.

    #correlation-analysis #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash uv run endopipe compute-noise-statistics -d ```

    To run the workflow for a single dataset:

    ```bash uv run endopipe compute-noise-statistics --datasets DATASET_NAME ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the noise
    correlations analysis for the first dataset, using only a subset of the
    timepoints and patches available in the dataset.

    Parameters
    ----------
    patch_type
        Patch type used to compute correlations.
    datasets
        List of datasets or dataset collections to compute correlations for.
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
        compute_noise_statistics_dataframe,
        get_trajectories_and_differences_for_noise_correlations,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
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
    from endo_pipeline.settings.autocorrelations import MAX_LAG_NOISE_COVARIANCE
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.manifest_names import (
        NOISE_CORRELATIONS_MANIFEST_NAME_PREFIXES,
        NOISE_PERIODOGRAM_MANIFEST_NAME_PREFIXES,
        NOISE_STATISTICS_MANIFEST_NAME_PREFIXES,
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

    max_num_timepoints = None
    max_num_patches = None
    if DEMO_MODE:
        max_num_timepoints = 25
        max_num_patches = 50
        dataset_names = dataset_names[:1]
        logger.warning(
            "DEMO MODE - Limiting to %d datasets, %d patches per dataset, "
            "and no more than %d steady state timepoints.",
            len(dataset_names),
            max_num_patches,
            max_num_timepoints,
        )

    # Default list of feature column names to use for correlation analysis if
    # not provided. Otherwise, use provided list.
    column_names = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load feature dataframe for specified patch type
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load pre-computed vector field dataframe manifest
    name_suffix = join_sorted_strings(column_names)
    vector_field_manifest_name = f"{VECTOR_FIELD_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)

    # Build dataframe manifest names that include sorted list of selected
    # columns used to compute noise statistics.
    name_prefix = NOISE_STATISTICS_MANIFEST_NAME_PREFIXES[patch_type]
    noise_statistics_manifest_name = f"{name_prefix}_{name_suffix}"
    noise_statistics_manifest = create_dataframe_manifest(
        noise_statistics_manifest_name, workflow_name=__file__
    )

    # Add parameters to dataframe manifest for traceability
    noise_statistics_manifest.parameters = {
        "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
        "run_name": DEFAULT_MODEL_RUN_NAME,
        "patch_type": patch_type,
        "columns": [f"{column}" for column in column_names],
    }
    save_dataframe_manifest(noise_statistics_manifest)

    # Build dataframe manifest names for the full two-timepoint noise
    # correlation functions and the residual periodograms computed alongside
    # the noise statistics dataframe.
    correlations_name_prefix = NOISE_CORRELATIONS_MANIFEST_NAME_PREFIXES[patch_type]
    noise_correlations_manifest_name = f"{correlations_name_prefix}_{name_suffix}"
    noise_correlations_manifest = create_dataframe_manifest(
        noise_correlations_manifest_name, workflow_name=__file__
    )
    noise_correlations_manifest.parameters = noise_statistics_manifest.parameters
    save_dataframe_manifest(noise_correlations_manifest)

    periodogram_name_prefix = NOISE_PERIODOGRAM_MANIFEST_NAME_PREFIXES[patch_type]
    noise_periodogram_manifest_name = f"{periodogram_name_prefix}_{name_suffix}"
    noise_periodogram_manifest = create_dataframe_manifest(
        noise_periodogram_manifest_name, workflow_name=__file__
    )
    noise_periodogram_manifest.parameters = noise_statistics_manifest.parameters
    save_dataframe_manifest(noise_periodogram_manifest)

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                feature_dataframe_manifest_name,
            )
            continue

        # load dataframe and filter to just steady state timepoints
        df_delayed = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_delayed[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # load vector field dataframe for the current dataset
        df_vec = load_dataframe(vector_field_manifest.locations[dataset_name], delay=False)

        # process on a per-flow condition basis
        noise_statistics_dataframe_list = []
        noise_correlation_function_dataframe_list = []
        periodogram_dataframe_list = []
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

            # Get callable vector field for the current flow condition
            vector_field_for_flow_condition = filter_dataframe_by_shear_stress(
                df_vec, flow_condition.shear_stress
            )
            vector_field_dict = get_vector_field_as_dict_from_dataframe(
                vector_field_for_flow_condition, column_names
            )
            vector_field = get_callable_vector_field(vector_field_dict, for_solve_ivp=False)

            # Create array for cross-correlation results at t vs. t' values, where t
            # and t' have range equal to the full range of timepoints present in
            # this flow condition
            timepoints_range = sorted(df_flow[Column.TIMEPOINT].unique())
            if max_num_timepoints is not None:
                timepoints_range = timepoints_range[-int(max_num_timepoints) :]
            logger.debug("Number of timepoints: %d", len(timepoints_range))
            timepoints_array = np.asarray(timepoints_range)
            # Need to re-mask the dataframe to include only the selected timepoints
            df_flow = df_flow[df_flow[Column.TIMEPOINT].isin(timepoints_array)]

            # Get the per-patch trajectories and their forward differences for
            # noise correlation analysis
            traj_list, d_traj_list = get_trajectories_and_differences_for_noise_correlations(
                df_flow, column_names, max_num_patches=max_num_patches
            )
            if not traj_list or not d_traj_list:
                logger.warning("No valid patches found for [ %s ]. Skipping.", dataset_name_flow)
                continue

            metadata_dict = {
                Column.DATASET: dataset_name,
                Column.SHEAR_STRESS: flow_condition.shear_stress,
            }

            # Compute noise correlation statistics (noise correlation and
            # measurement noise variance per feature, as a function of lag) for
            # the current flow condition
            noise_statistics_dataframe = compute_noise_statistics_dataframe(
                traj_list=traj_list,
                d_traj_list=d_traj_list,
                column_names=column_names,
                timepoints_array=timepoints_array,
                vector_field=vector_field,
                max_lag=MAX_LAG_NOISE_COVARIANCE,
                metadata_dict=metadata_dict,
            )
            noise_statistics_dataframe_list.append(noise_statistics_dataframe)

        if not noise_statistics_dataframe_list:
            logger.warning("No noise statistics computed for [ %s ]. Skipping.", dataset_name)
            continue

        # Concatenate noise statistics dataframes for each flow condition
        noise_statistics_for_dataset = pd.concat(noise_statistics_dataframe_list, ignore_index=True)

        # Save dataframe to file
        save_path = output_path / f"{name_prefix}_{dataset_name}_{name_suffix}.parquet"
        noise_statistics_for_dataset.to_parquet(save_path, index=False)

        # Create location object with output path
        location = noise_statistics_manifest.locations.get(dataset_name, DataframeLocation())
        location.path = save_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(
                dataset_config,
                model_manifest=load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME),
                run_name=DEFAULT_MODEL_RUN_NAME,
                additional_notes="Dataframe containing noise correlation statistics for various features.",
            )
            fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None

        # Add dataframe location to dataframe manifest and save
        noise_statistics_manifest.locations[dataset_name] = location
        save_dataframe_manifest(noise_statistics_manifest)

        # Save the two-timepoint noise correlation functions for the dataset
        if noise_correlation_function_dataframe_list:
            noise_correlation_function_for_dataset = pd.concat(
                noise_correlation_function_dataframe_list, ignore_index=True
            )
            correlations_save_path = (
                output_path / f"{correlations_name_prefix}_{dataset_name}_{name_suffix}.parquet"
            )
            noise_correlation_function_for_dataset.to_parquet(correlations_save_path, index=False)

            correlations_location = noise_correlations_manifest.locations.get(
                dataset_name, DataframeLocation()
            )
            correlations_location.path = correlations_save_path
            if UPLOAD_TO_FMS:
                fmsid = upload_file_to_fms(
                    correlations_save_path, annotations=annotations, file_type="parquet"
                )
                correlations_location.fmsid = fmsid
                correlations_location.path = None
            noise_correlations_manifest.locations[dataset_name] = correlations_location
            save_dataframe_manifest(noise_correlations_manifest)

        # Save the residual periodograms for the dataset
        if periodogram_dataframe_list:
            periodogram_for_dataset = pd.concat(periodogram_dataframe_list, ignore_index=True)
            periodogram_save_path = (
                output_path / f"{periodogram_name_prefix}_{dataset_name}_{name_suffix}.parquet"
            )
            periodogram_for_dataset.to_parquet(periodogram_save_path, index=False)

            periodogram_location = noise_periodogram_manifest.locations.get(
                dataset_name, DataframeLocation()
            )
            periodogram_location.path = periodogram_save_path
            if UPLOAD_TO_FMS:
                fmsid = upload_file_to_fms(
                    periodogram_save_path, annotations=annotations, file_type="parquet"
                )
                periodogram_location.fmsid = fmsid
                periodogram_location.path = None
            noise_periodogram_manifest.locations[dataset_name] = periodogram_location
            save_dataframe_manifest(noise_periodogram_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
