from endo_pipeline.cli import Datasets, PatchType
from endo_pipeline.library.analyze.numerics.correlations import NoiseCorrelationCentering


def main(patch_type: PatchType = "grid_based", datasets: Datasets | None = None) -> None:
    """
    Compute noise statistics from feature displacements at steady state.

    This analysis serves as a validation of whether or not the noise term of the
    underlying dynamics is indeed uncorrelated across timepoints (white noise).

    #correlation-analysis #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-noise-statistics -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-noise-statistics --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    noise correlations analysis for the first dataset, using only a subset
    of the timepoints and patches available in the dataset.

    Parameters
    ----------
    patch_type
        Patch type used to compute correlations.
    datasets
        List of datasets or dataset collections to compute correlations for.
    """
    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import join_sorted_strings, load_dataframe
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
        compute_lagged_covariances,
        get_trajectories_and_differences_for_noise_correlations,
        normalize_lagged_covariances,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_vector_field_as_dict_from_dataframe,
    )
    from endo_pipeline.library.analyze.vector_field_function import get_callable_vector_field
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.manifest_names import VECTOR_FIELD_MANIFEST_NAMES
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

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
        for flow_condition in dataset_config.flow_conditions:
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"
            print(dataset_name_flow, "\n")

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

            # Compute covariances as a function of lag for the current centering method
            for centering_method in NoiseCorrelationCentering:
                lagged_covariances = compute_lagged_covariances(
                    traj_list=traj_list,
                    d_traj_list=d_traj_list,
                    column_names=column_names,
                    timepoints_array=timepoints_array,
                    vector_field=vector_field,
                    centering_method=centering_method,
                    max_lag=15,
                )
                lagged_correlations = normalize_lagged_covariances(lagged_covariances)

                lag_zero_covariance = np.einsum("ii->i", lagged_covariances[0])
                print(lag_zero_covariance)
                lag_one_covariance = np.einsum("ii->i", lagged_covariances[1])
                print(lag_one_covariance)
                measurement_noise_variance = np.where(
                    lag_one_covariance < 0, lag_one_covariance, np.nan
                )
                print(measurement_noise_variance)
                lag_gt_one_correlation = np.einsum("...ii->...i", lagged_correlations[2:])
                print(np.max(np.abs(lag_gt_one_correlation)))
                print(np.min(np.abs(lag_gt_one_correlation)))
                print(np.mean(lag_gt_one_correlation), "\n")
