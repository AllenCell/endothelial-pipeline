# %%
import logging

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe, save_plot_to_path
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

# %%
logger = logging.getLogger(__name__)

output_path = get_output_path("noise_correlations")

dataset_names = [
    *get_datasets_in_collection("shear_stress"),
    *get_datasets_in_collection("perturbation"),
]

patch_type = "grid_based"

max_num_timepoints = None
max_num_patches = None
if DEMO_MODE:
    max_num_timepoints = 50
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
# %%
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

    df_vec = load_dataframe(vector_field_manifest.locations[dataset_name], delay=False)

    # process on a per-flow condition basis
    cross_corr_dataframe_list = []
    for flow_condition in dataset_config.flow_conditions:
        shear_stress = flow_condition.shear_stress
        dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"

        df_flow = filter_dataframe_to_flow_condition_by_timepoint(
            df_steady_state, dataset_config, flow_condition
        )
        metadata_dict = {
            Column.DATASET: dataset_name,
            Column.SHEAR_STRESS: shear_stress,
        }
        steady_state_duration = df_flow[Column.TIMEPOINT].max() - df_flow[Column.TIMEPOINT].min()
        track_duration_filter = min(LONG_TRACK_THRESHOLD_LENGTH, steady_state_duration)
        df_flow = add_track_duration_to_dataframe(
            df_flow, grouping_columns=[Column.CROP_INDEX], time_column=Column.TIMEPOINT
        )
        df_flow = filter_dataframe_by_track_length(
            dataframe=df_flow, minimum_track_length=track_duration_filter
        )

        vector_field_for_flow_condition = filter_dataframe_by_shear_stress(df_vec, shear_stress)

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
        cross_corr_results = np.zeros((len(timepoints_range), len(timepoints_range), n_dim, n_dim))
        n_points = np.zeros((len(timepoints_range), len(timepoints_range)))

        # Loop over each patch in the dataset and compute
        num_patches_processed = 0
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
            mean_dx_t = filtered_d_traj[diff_column_names].mean(axis=0).to_numpy()

            # if either the returned trajectory or difference arrays are empty, skip
            # this trajectory
            if filtered_traj.empty or filtered_d_traj.empty:
                logger.warning(
                    "Skipping patch with empty trajectory or difference arrays: [ %s ]", patch_idx
                )
                continue
            num_patches_processed += 1

            for t_idx, t in enumerate(timepoints_range):
                dx_t = filtered_d_traj[filtered_d_traj[Column.TIMEPOINT] == t][
                    diff_column_names
                ].to_numpy()
                if len(dx_t) == 0:
                    continue
                for t_prime_idx, t_prime in enumerate(timepoints_range):
                    # Compute cross-correlation for this pair of timepoints
                    dx_t_prime = filtered_d_traj[filtered_d_traj[Column.TIMEPOINT] == t_prime][
                        diff_column_names
                    ].to_numpy()
                    if len(dx_t_prime) == 0:
                        continue

                    # R_ij(t, t') = < eta_i(t) eta_j(t') >
                    cross_corr_results[t_idx, t_prime_idx] += np.outer(
                        (dx_t - mean_dx_t) / TIME_STEP_IN_HOURS,
                        (dx_t_prime - mean_dx_t) / TIME_STEP_IN_HOURS,
                    )
                    n_points[t_idx, t_prime_idx] += 1

        # Normalize accumulated sums by the number of contributing patches
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_corr_results /= n_points[:, :, np.newaxis, np.newaxis]
        cross_corr_results = np.nan_to_num(cross_corr_results, nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(n_dim):
            for j in range(n_dim):
                # Plot the cross-correlation matrix: R_ij(t, t')
                # where the x axis is t and the y axis is t'
                fig, ax = plt.subplots(figsize=(8, 6))
                cax = ax.pcolormesh(
                    cross_corr_results[:, :, i, j].T,
                    cmap="coolwarm",
                    norm=colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1),
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

                # Scatter plot of R_ij(t, t') as a function of
                # tau = |t - t'|
                fig, ax = plt.subplots(figsize=(8, 6))
                tau = np.abs(np.subtract.outer(timepoints_range, timepoints_range))
                cross_corr_flat = cross_corr_results[:, :, i, j].flatten()
                tau_flat = tau.flatten()
                ax.scatter(tau_flat, cross_corr_flat, alpha=0.5)
                ax.set_xlabel("$\\tau = |t - t'|$")
                ax.set_ylabel("$R_{ij}(t, t')$")
                ax.set_title(
                    f"Cross-Correlation vs Time Lag: R_ij(t, t') for (i,j) = ({column_names[i]}, {column_names[j]})"
                )
                figure_name = f"noise_correlation_vs_tau_{dataset_name_flow}_cols_{column_names[i]}_{column_names[j]}"
                save_plot_to_path(fig, output_path, figure_name)

# %%
