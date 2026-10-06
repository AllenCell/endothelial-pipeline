from endo_pipeline.cli import Datasets, PatchType


def main(patch_type: PatchType = "grid_based", datasets: Datasets | None = None) -> None:
    """
    Compute two-timepoint cross correlation of feature displacements at steady state.

    This analysis serves as a validation of whether or not the noise term of the
    underlying dynamics is indeed uncorrelated across timepoints (white noise).

    #correlation-analysis #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-noise-correlations -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-noise-correlations --datasets DATASET_NAME
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
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_shear_stress,
        filter_dataframe_by_track_length,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_track_duration_to_dataframe,
    )
    from endo_pipeline.library.analyze.numerics.forward_difference import (
        compute_forward_differences_along_trajectory,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_vector_field_as_dict_from_dataframe,
    )
    from endo_pipeline.library.analyze.vector_field_function import get_callable_vector_field
    from endo_pipeline.library.visualize.diffae_features.noise_correlations import (
        plot_cross_correlations_against_lag,
        plot_noise_amplitude,
        plot_normalized_two_timepoint_cross_correlations,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
        TIME_STEP_IN_HOURS,
    )
    from endo_pipeline.settings.manifest_names import VECTOR_FIELD_MANIFEST_NAMES
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

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
    diff_column_names = [f"{col}{Column.DiffAEData.DIFFERENCE_SUFFIX}" for col in column_names]
    n_dim = len(column_names)
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

            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            steady_state_duration = (
                df_flow[Column.TIMEPOINT].max() - df_flow[Column.TIMEPOINT].min()
            )
            track_duration_filter = min(LONG_TRACK_THRESHOLD_LENGTH, steady_state_duration)
            df_flow = add_track_duration_to_dataframe(
                df_flow, grouping_columns=[Column.CROP_INDEX], time_column=Column.TIMEPOINT
            )
            df_flow = filter_dataframe_by_track_length(
                dataframe=df_flow, minimum_track_length=track_duration_filter
            )

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
            n_timepoints = len(timepoints_range)
            timepoints_array = np.asarray(timepoints_range)
            df_flow = df_flow[df_flow[Column.TIMEPOINT].isin(timepoints_array)]

            cross_corr_mean_subtracted = np.zeros((n_timepoints, n_timepoints, n_dim, n_dim))
            cross_corr_drift_subtracted = np.zeros((n_timepoints, n_timepoints, n_dim, n_dim))
            n_points_ms = np.zeros((n_timepoints, n_timepoints))
            n_points_ds = np.zeros((n_timepoints, n_timepoints))

            # Loop over each patch in the dataset and compute the forward differences
            num_patches_processed = 0
            traj_list = []
            d_traj_list = []
            for patch_idx, df_patch_ in df_flow.groupby(Column.CROP_INDEX):
                if max_num_patches is not None and num_patches_processed >= max_num_patches:
                    logger.debug("Reached maximum number of patches: %d", max_num_patches)
                    break
                logger.debug("Processing patch: %s", patch_idx)
                # forward differences require at least two distinct timepoints
                if df_patch_[Column.TIMEPOINT].nunique() < 2:
                    logger.warning("Skipping patch with insufficient timepoints: [ %s ]", patch_idx)
                    continue

                # sort by timepoint to ensure that trajectory is in correct order before
                # computing differences
                df_patch = df_patch_.sort_values(by=Column.TIMEPOINT)

                # compute forward differences along trajectory for this crop, and filter
                # to keep only differences between timepoints that are separated by
                # time_lag number of frames (accounts for any missing timepoints in the
                # trajectory, for example due to outlier filtering)
                filtered_traj, filtered_d_traj = compute_forward_differences_along_trajectory(
                    df_patch, column_names
                )

                # if the returned difference array is empty, skip this trajectory
                if filtered_traj.empty or filtered_d_traj.empty:
                    logger.warning(
                        "Skipping patch with empty trajectory or difference arrays: [ %s ]",
                        patch_idx,
                    )
                    continue

                traj_list.append(filtered_traj)
                d_traj_list.append(filtered_d_traj)
                num_patches_processed += 1

            if not traj_list or not d_traj_list:
                logger.warning("No valid patches found for [ %s ]. Skipping.", dataset_name_flow)
                continue

            # Ensemble mean of the forward differences across patches at each timepoint
            mean_dx_t = (
                pd.concat(d_traj_list)
                .groupby(Column.TIMEPOINT)[diff_column_names]
                .mean()
                .reindex(timepoints_range)
                .to_numpy()
            )

            for filtered_traj, filtered_d_traj in zip(traj_list, d_traj_list, strict=True):
                # Extract the timepoints, values, and differences for this patch
                patch_timepoints_x_t = filtered_traj[Column.TIMEPOINT].to_numpy()
                patch_timepoints_dx_t = filtered_d_traj[Column.TIMEPOINT].to_numpy()
                patch_x_t = filtered_traj[column_names].to_numpy()
                patch_dx_t = filtered_d_traj[diff_column_names].to_numpy()

                # place this patch's trajectory values and differences on the
                # shared timepoint axis, leaving NaN at timepoints where the
                # patch has no difference/value
                x_t = np.full((n_timepoints, n_dim), np.nan)
                x_t[np.searchsorted(timepoints_array, patch_timepoints_x_t)] = patch_x_t
                dx_t = np.full((n_timepoints, n_dim), np.nan)
                dx_t[np.searchsorted(timepoints_array, patch_timepoints_dx_t)] = patch_dx_t
                f_x_t = vector_field(x_t)

                # mean subtracted differences
                eta_ms = dx_t - mean_dx_t
                # drift subtracted differences
                eta_ds = dx_t - f_x_t * TIME_STEP_IN_HOURS
                # mask valid timepoints for centered differences
                is_valid_ms = ~np.isnan(eta_ms).any(axis=1)
                eta_ms = np.where(is_valid_ms[:, np.newaxis], eta_ms, 0.0)
                is_valid_ds = ~np.isnan(eta_ds).any(axis=1)
                eta_ds = np.where(is_valid_ds[:, np.newaxis], eta_ds, 0.0)

                # R_ij(t, t') = < eta_i(t) eta_j(t') >
                cross_corr_mean_subtracted += np.einsum("ti,sj->tsij", eta_ms, eta_ms)
                n_points_ms += np.outer(is_valid_ms, is_valid_ms)
                cross_corr_drift_subtracted += np.einsum("ti,sj->tsij", eta_ds, eta_ds)
                n_points_ds += np.outer(is_valid_ds, is_valid_ds)

            # Normalize accumulated sums by the number of contributing patches
            with np.errstate(divide="ignore", invalid="ignore"):
                cross_corr_mean_subtracted /= n_points_ms[:, :, np.newaxis, np.newaxis]
                cross_corr_drift_subtracted /= n_points_ds[:, :, np.newaxis, np.newaxis]

            # make plots
            for cross_corr_centered, subtracted in [
                (cross_corr_mean_subtracted, "mean subtracted"),
                (cross_corr_drift_subtracted, "drift subtracted"),
            ]:
                # sigma_i(t) = sqrt(R_ii(t, t)); stationary noise has a flat sigma_i(t)
                diagonal_indices = np.arange(n_timepoints)
                variance_t = cross_corr_centered[diagonal_indices, diagonal_indices]
                sigma_t = np.sqrt(np.einsum("tii->ti", variance_t))
                _ = plot_noise_amplitude(
                    sigma_t=sigma_t,
                    timepoints_array=timepoints_array,
                    column_names=column_names,
                    plot_title=f"Noise amplitude ({subtracted}): {dataset_name_flow}",
                    output_path=output_path,
                    file_name=f"noise_amplitude_{subtracted}_{dataset_name_flow}",
                )

                # rho_ij(t, t') = R_ij(t, t') / (sigma_i(t) * sigma_j(t')),
                # which rescales to [-1, 1] and divides out any potential drift
                # in the noise amplitude
                plot_normalized_two_timepoint_cross_correlations(
                    cross_correlations=cross_corr_centered,
                    sigma_t=sigma_t,
                    timepoints_range=timepoints_range,
                    column_names=column_names,
                    plot_title=f"Normalized cross-correlations ({subtracted}): {dataset_name_flow}",
                    output_path=output_path,
                    file_name=f"normalized_cross_correlations_{subtracted}_{dataset_name_flow}",
                )

                # Plot R_ij(t,t') = R_ji(t',t) as a function of the time lag tau = t' - t
                plot_cross_correlations_against_lag(
                    cross_correlations=cross_corr_centered,
                    timepoints_array=timepoints_array,
                    column_names=column_names,
                    plot_title=f"Cross-correlations vs time lag ({subtracted}): {dataset_name_flow}",
                    output_path=output_path,
                    file_name=f"cross_correlations_vs_lag_{subtracted}_{dataset_name_flow}",
                )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
