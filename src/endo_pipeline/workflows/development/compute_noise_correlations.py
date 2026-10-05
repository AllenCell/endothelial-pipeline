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

    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_filtering import (
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
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
    )
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
            cross_corr_results = np.zeros((n_timepoints, n_timepoints, n_dim, n_dim))
            n_points = np.zeros((n_timepoints, n_timepoints))

            # Loop over each patch in the dataset and compute the forward differences
            num_patches_processed = 0
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
                _, filtered_d_traj = compute_forward_differences_along_trajectory(
                    df_patch, column_names
                )

                # if the returned difference array is empty, skip this trajectory
                if filtered_d_traj.empty:
                    logger.warning(
                        "Skipping patch with empty difference arrays: [ %s ]",
                        patch_idx,
                    )
                    continue

                d_traj_list.append(filtered_d_traj)
                num_patches_processed += 1

            if not d_traj_list:
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

            for filtered_d_traj in d_traj_list:
                # Extract the timepoints and differences for this patch
                patch_timepoints = filtered_d_traj[Column.TIMEPOINT].to_numpy()
                patch_dx_t = filtered_d_traj[diff_column_names].to_numpy()

                # place this patch's differences on the shared timepoint axis,
                # leaving NaN at timepoints where the patch has no difference
                dx_t = np.full((n_timepoints, n_dim), np.nan)
                dx_t[np.searchsorted(timepoints_array, patch_timepoints)] = patch_dx_t

                # mean subtracted differences
                eta = dx_t - mean_dx_t
                is_valid = ~np.isnan(eta).any(axis=1)
                eta = np.where(is_valid[:, np.newaxis], eta, 0.0)

                # R_ij(t, t') = < eta_i(t) eta_j(t') >
                cross_corr_results += np.einsum("ti,sj->tsij", eta, eta)
                n_points += np.outer(is_valid, is_valid)

            # Normalize accumulated sums by the number of contributing patches
            with np.errstate(divide="ignore", invalid="ignore"):
                cross_corr_results /= n_points[:, :, np.newaxis, np.newaxis]

            # R_ij(t, t') = R_ji(t', t), so only the unique feature pairs are plotted
            for i in range(n_dim):
                for j in range(i, n_dim):
                    # Plot the cross-correlation matrix: R_ij(t, t')
                    # where the x axis is t and the y axis is t'
                    corr_matrix = cross_corr_results[:, :, i, j]
                    abs_max = np.nanmax(np.abs(corr_matrix))
                    if not np.isfinite(abs_max) or abs_max == 0:
                        abs_max = 1.0
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cax = ax.pcolormesh(
                        corr_matrix.T,
                        cmap="coolwarm",
                        norm=colors.TwoSlopeNorm(vcenter=0, vmin=-0.01, vmax=0.01),
                        shading="auto",
                    )
                    fig.colorbar(cax)
                    ax.set_xlabel("Timepoint $t$")
                    ax.set_xticks(
                        np.arange(len(timepoints_range))[::10] + 0.5, labels=timepoints_range[::10]
                    )
                    ax.set_ylabel("Timepoint $t'$")
                    ax.set_yticks(
                        np.arange(len(timepoints_range))[::10] + 0.5, labels=timepoints_range[::10]
                    )
                    ax.set_title(
                        f"Cross-Correlation Matrix: R_ij(t, t') for (i,j) = ({column_names[i]}, {column_names[j]})"
                    )
                    figure_name = f"noise_correlation_matrix_{dataset_name_flow}_cols_{column_names[i]}_{column_names[j]}"
                    save_plot_to_path(fig, output_path, figure_name)

                    # Scatter plot of R_ij(t, t') as a function of the signed lag
                    # tau = t' - t, where the tau < 0 half carries R_ji
                    fig, ax = plt.subplots(figsize=(8, 6))
                    tau = timepoints_array[np.newaxis, :] - timepoints_array[:, np.newaxis]
                    # R_ii is symmetric in tau, so keep only non-negative lags
                    keep = tau >= 0 if i == j else np.ones_like(tau, dtype=bool)
                    ax.scatter(tau[keep], corr_matrix[keep], alpha=0.5)
                    ax.set_xlabel("$\\tau = t' - t$")
                    ax.set_ylabel("$R_{ij}(t, t')$")
                    ax.set_title(
                        f"Cross-Correlation vs Time Lag: R_ij(t, t') for (i,j) = ({column_names[i]}, {column_names[j]})"
                    )
                    figure_name = f"noise_correlation_vs_tau_{dataset_name_flow}_cols_{column_names[i]}_{column_names[j]}"
                    save_plot_to_path(fig, output_path, figure_name)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
