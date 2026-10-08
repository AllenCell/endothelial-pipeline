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

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        save_plot_to_path,
        slugify,
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
        compute_two_timepoint_noise_correlations,
        get_trajectories_and_differences_for_noise_correlations,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_vector_field_as_dict_from_dataframe,
    )
    from endo_pipeline.library.analyze.vector_field_function import get_callable_vector_field
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
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
    n_dim = len(column_names)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load feature dataframe for specified patch type
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load pre-computed vector field dataframe manifest
    name_suffix = join_sorted_strings(column_names)
    vector_field_manifest_name = f"{VECTOR_FIELD_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)

    # Lags to loop over in units frames
    dt_values = np.arange(1, 25)

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
            n_timepoints = len(timepoints_range)
            timepoints_array = np.asarray(timepoints_range)
            # Need to re-mask the dataframe to include only the selected timepoints
            df_flow = df_flow[df_flow[Column.TIMEPOINT].isin(timepoints_array)]

            # Compute MSD and make plots
            for centering_method in NoiseCorrelationCentering:
                # MSD tensor for each time lag dt
                msd_values = np.zeros((len(dt_values), n_dim, n_dim))
                for dt_index, dt in enumerate(dt_values):
                    traj_list, d_traj_list = (
                        get_trajectories_and_differences_for_noise_correlations(
                            df_flow, column_names, max_num_patches=max_num_patches, time_lag=dt
                        )
                    )
                    if not traj_list or not d_traj_list:
                        logger.warning(
                            "No valid patches found for [ %s ]. Skipping.", dataset_name_flow
                        )
                        continue

                    # Compute cross-correlations for the current centering method
                    cross_correlations = compute_two_timepoint_noise_correlations(
                        traj_list=traj_list,
                        d_traj_list=d_traj_list,
                        column_names=column_names,
                        timepoints_array=timepoints_array,
                        vector_field=vector_field,
                        centering_method=centering_method,
                    )

                    if cross_correlations is None:
                        # If method returns early due to an error, continue
                        continue
                    else:
                        # MSD_ij = <R_ij(t, t)> averaged over time (diagonal elements of the cross-correlation matrix),
                        # where i, j = 0, 1, ..., n_dim-1
                        diagonal_indices = np.arange(n_timepoints)
                        variance_t = cross_correlations[diagonal_indices, diagonal_indices]
                        msd_values[dt_index] = np.nanmean(variance_t, axis=0)

                # Plot MSD vs. dt for the current centering method on a log-log scale,
                # and fit MSD(dt) = (dt)^alpha to extract the scaling exponent alpha.
                for i in range(n_dim):
                    for j in range(i, n_dim):
                        msd_ij = msd_values[:, i, j]
                        where_finite = ~np.isnan(msd_ij)
                        where_approx_zero = np.abs(msd_ij) < 1e-3
                        msd_valid = msd_ij[where_finite][~where_approx_zero]
                        dt_valid = dt_values[where_finite][~where_approx_zero]
                        log_msd = np.log(msd_valid)
                        log_dt = np.log(dt_valid)
                        alpha, intercept = np.polyfit(log_dt, log_msd, 1)
                        predicted_line = alpha * log_dt + intercept
                        fit_coeff_determination = 1 - np.sum(
                            (log_msd - predicted_line) ** 2
                        ) / np.sum((log_msd - np.mean(log_msd)) ** 2)
                        fig, ax = plt.subplots()
                        ax.loglog(dt_values, msd_ij, color="k", marker="o", label=f"MSD_{i}{j}")
                        ax.loglog(
                            dt_valid,
                            np.exp(predicted_line),
                            color="b",
                            linestyle="--",
                            label=f"Fit alpha={alpha:.2f} ($R^2 = ${fit_coeff_determination:.3f})",
                        )
                        ax.legend()
                        ax.set_xlabel("Time lag (dt)")
                        ax.set_ylabel("MSD")
                        ax.set_ylim((5e-4, 2e-1))
                        ax.set_title(f"MSD vs. dt ({centering_method})")
                        column_label_i = COLUMN_METADATA[column_names[i]].label or column_names[i]
                        column_label_j = COLUMN_METADATA[column_names[j]].label or column_names[j]
                        plot_title = f"MSD vs. dt ({centering_method}): {dataset_name_flow}"
                        ax.set_title(
                            f"{plot_title}\nMSD$_{{ij}}(dt)$ for $(i,j)$ = ({column_label_i}, {column_label_j})"
                        )
                        figure_name = slugify(f"msd_{centering_method}_{dataset_name_flow}")
                        figure_name = f"{figure_name}_{column_names[i]}_{column_names[j]}"
                        save_plot_to_path(fig, output_path, figure_name)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
