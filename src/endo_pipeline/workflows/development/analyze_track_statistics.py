from endo_pipeline.cli import Datasets
from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH


def main(
    datasets: Datasets | None = None,
    min_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
) -> None:
    import logging
    from typing import cast

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import circmean, circvar

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_track_length,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.track_statistics import (
        compute_kde_on_bins,
        plot_kde,
        smooth_kde_with_spline,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.bootstrap_fixed_points import (
        FP_CI_LOWER_PERCENTILE,
        FP_CI_UPPER_PERCENTILE,
        NUM_BOOTSTRAP_ITERATIONS,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        POLAR_ANGLE_PERIOD,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(RANDOM_SEED)

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names: list[Column.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)
    variable_labels_dict = {col: get_label_for_column(col) for col in column_names}
    columns_to_compute_grid = [*METADATA_COLUMNS_TO_KEEP["grid"], *column_names]
    columns_to_compute_tracked = [*METADATA_COLUMNS_TO_KEEP["tracked"], *column_names]

    # kernel names for KDEs
    kernel_names_dict = cast(dict[str | Column.DiffAEData, str], KERNEL_NAMES_DYNAMICS.copy())

    # bin widths for histograms of column averages and variances across
    # trajectories (currently hardcoded)
    bin_width_averages = 0.1
    bin_width_variances = 0.02
    kde_eval_num_points = 2000

    # Get dataframe manifest for filtered crop-based features
    base_name_grid = f"{model_manifest_name}_{run_name}_grid"
    grid_feature_dataframe_manifest_name = f"{base_name_grid}_pca_filtered"
    grid_feature_dataframe_manifest = load_dataframe_manifest(grid_feature_dataframe_manifest_name)
    base_name_tracked = f"{model_manifest_name}_{run_name}_tracked"
    tracked_feature_dataframe_manifest_name = f"{base_name_tracked}_pca_filtered"
    tracked_feature_dataframe_manifest = load_dataframe_manifest(
        tracked_feature_dataframe_manifest_name
    )

    # Default list of datasets if not provided. Filter by datasets available in
    # the manifest.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    # bin limits and polar angle period are constant
    bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    polar_angle_period = POLAR_ANGLE_PERIOD

    for dataset_name in dataset_names:
        if (
            dataset_name not in grid_feature_dataframe_manifest.locations
            or dataset_name not in tracked_feature_dataframe_manifest.locations
        ):
            logger.warning(
                "No feature dataframe found in manifest [ %s ] or [ %s ] for dataset [ %s ]. Skipping this dataset.",
                grid_feature_dataframe_manifest_name,
                tracked_feature_dataframe_manifest_name,
                dataset_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.shear_stress_regime) > 1:
            logger.warning(
                "Dataset [ %s ] has more than one shear stress condition: [ %s ]. Skipping this dataset.",
                dataset_name,
                dataset_config.shear_stress_regime,
            )
            continue

        shear_stress = dataset_config.flow_conditions[0].shear_stress
        dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
        plot_label = f"{dataset_name} ({shear_stress} dyn/cm$^2$)"
        fig_savedir = get_output_path(__file__, dataset_name)

        # load dataframe and perform additional filtering (e.g., remove
        # non-steady-state timepoints based on annotations), computing only the
        # columns needed for analysis
        dataset_config = load_dataset_config(dataset_name)
        df_grid_ = load_dataframe(
            grid_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_grid: pd.DataFrame = df_grid_[columns_to_compute_grid].compute()
        df_steady_state_grid = filter_dataframe_to_steady_state(df_grid, dataset_config)
        num_trajectories_grid = df_steady_state_grid[Column.CROP_INDEX].nunique()
        df_steady_state_grid[Column.TRACK_LENGTH] = df_steady_state_grid.groupby(Column.CROP_INDEX)[
            Column.TIMEPOINT
        ].transform(lambda t: t.max() - t.min())
        # Perform additional filtering by track length
        df_steady_state_grid = filter_dataframe_by_track_length(
            df_steady_state_grid, min_track_length
        )

        df_tracked_ = load_dataframe(
            tracked_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_tracked: pd.DataFrame = df_tracked_[columns_to_compute_tracked].compute()
        df_steady_state_tracked = filter_dataframe_to_steady_state(df_tracked, dataset_config)
        df_steady_state_tracked[Column.TRACK_LENGTH] = df_steady_state_tracked.groupby(
            Column.CROP_INDEX
        )[Column.TIMEPOINT].transform(lambda t: t.max() - t.min())
        # Perform additional filtering by track length
        df_steady_state_tracked = filter_dataframe_by_track_length(
            df_steady_state_tracked, min_track_length
        )
        num_trajectories_tracked = df_steady_state_tracked[Column.CROP_INDEX].nunique()

        logger.info(
            "Dataset [ %s ] has %d grid trajectories and %d tracked trajectories. "
            "Bootstrap resampling tracked trajectories to match number of grid trajectories for comparison.",
            dataset_name,
            num_trajectories_grid,
            num_trajectories_tracked,
        )

        # put together grid and tracked dataframes for easier processing, adding
        # a column to indicate crop pattern
        df_steady_state_dict: dict[str, pd.DataFrame] = {
            "grid": df_steady_state_grid,
            "tracked": df_steady_state_tracked,
        }

        base_df = pd.DataFrame(columns=[Column.CROP_INDEX, *column_names])
        column_avg_df_dict: dict[str, pd.DataFrame] = {
            "grid": base_df.copy(),
            "tracked": base_df.copy(),
        }
        column_variance_df_dict: dict[str, pd.DataFrame] = {
            "grid": base_df.copy(),
            "tracked": base_df.copy(),
        }
        x_eval_avg_dict: dict = {"grid": {}, "tracked": {}}
        x_eval_var_dict: dict = {"grid": {}, "tracked": {}}
        bins_avg_dict: dict = {"grid": {}, "tracked": {}}
        bins_var_dict: dict = {"grid": {}, "tracked": {}}
        for crop_pattern in ["grid", "tracked"]:
            for traj_index, df_traj in df_steady_state_dict[crop_pattern].groupby(
                Column.CROP_INDEX
            ):
                for column_name in column_names:
                    if column_name == Column.DiffAEData.POLAR_ANGLE:
                        # take circular mean for polar angle to account for periodicity
                        column_avg_df_dict[crop_pattern].loc[traj_index, column_name] = circmean(
                            df_traj[column_name],
                            high=bin_limits_dict[Column.DiffAEData.POLAR_ANGLE][1],
                            low=bin_limits_dict[Column.DiffAEData.POLAR_ANGLE][0],
                        )
                        column_variance_df_dict[crop_pattern].loc[traj_index, column_name] = (
                            circvar(
                                df_traj[column_name],
                                high=bin_limits_dict[Column.DiffAEData.POLAR_ANGLE][1],
                                low=bin_limits_dict[Column.DiffAEData.POLAR_ANGLE][0],
                            )
                        )
                    else:
                        column_avg_df_dict[crop_pattern].loc[traj_index, column_name] = np.nanmean(
                            df_traj[column_name]
                        )
                        column_variance_df_dict[crop_pattern].loc[traj_index, column_name] = (
                            np.nanvar(df_traj[column_name])
                        )

            # compute data-based bin evaluation grids for this crop pattern
            for column_name in column_names:
                avg_data = (
                    column_avg_df_dict[crop_pattern][column_name].dropna().to_numpy().reshape(-1, 1)
                )
                avg_bins = get_bins(bin_widths=(bin_width_averages,), data=avg_data)[0]
                bins_avg_dict[crop_pattern][column_name] = avg_bins[0]
                x_eval_avg_dict[crop_pattern][column_name] = np.linspace(
                    avg_bins[0][0], avg_bins[0][-1], kde_eval_num_points
                )

                var_data = (
                    column_variance_df_dict[crop_pattern][column_name]
                    .dropna()
                    .to_numpy()
                    .reshape(-1, 1)
                )
                var_bins = get_bins(bin_widths=(bin_width_variances,), data=var_data)[0]
                bins_var_dict[crop_pattern][column_name] = var_bins[0]
                x_eval_var_dict[crop_pattern][column_name] = np.linspace(
                    var_bins[0][0], var_bins[0][-1], kde_eval_num_points
                )

        # Bootstrap KDE computation for tracked crops: resample (with replacement)
        # tracked trajectories to the same size as grid trajectories, compute the
        # KDE for each bootstrap sample on the non-interpolated bin-center grid,
        # then report mean and (5, 95) CI on that same coarse grid.  Spline
        # smoothing to the fine x_eval grid is deferred to the plotting step.
        grid_avg_kde_dict: dict = {}
        grid_var_kde_dict: dict = {}
        bootstrap_tracked_avg_kde: dict = {}
        bootstrap_tracked_var_kde: dict = {}
        for column_name in column_names:
            period = polar_angle_period if column_name == Column.DiffAEData.POLAR_ANGLE else None

            # compute grid KDEs on the native bin-center grid
            grid_avg_bin_centers, grid_avg_kde_values = compute_kde_on_bins(
                data=column_avg_df_dict["grid"][column_name].dropna().to_numpy(),
                bin_width=bin_width_averages,
                kernel_name=kernel_names_dict[column_name],
                kernel_bandwidth=1.5 * bin_width_averages,
                kernel_period=period,
                bins=bins_avg_dict["grid"][column_name],
            )
            grid_avg_kde_dict[column_name] = {
                "bin_centers": grid_avg_bin_centers,
                "kde_values": grid_avg_kde_values,
            }
            grid_var_bin_centers, grid_var_kde_values = compute_kde_on_bins(
                data=column_variance_df_dict["grid"][column_name].dropna().to_numpy(),
                bin_width=bin_width_variances,
                kernel_name="gaussian",
                kernel_bandwidth=1.5 * bin_width_variances,
                kernel_period=None,
                bins=bins_var_dict["grid"][column_name],
            )
            grid_var_kde_dict[column_name] = {
                "bin_centers": grid_var_bin_centers,
                "kde_values": grid_var_kde_values,
            }

            tracked_avg_all = column_avg_df_dict["tracked"][column_name].dropna().to_numpy()
            tracked_var_all = column_variance_df_dict["tracked"][column_name].dropna().to_numpy()

            # use fixed bins derived from the full tracked data so all bootstrap
            # samples share the same bin-center grid and can be stacked directly
            fixed_avg_bins = bins_avg_dict["tracked"][column_name]
            fixed_var_bins = bins_var_dict["tracked"][column_name]
            tracked_avg_bin_centers = (fixed_avg_bins[:-1] + fixed_avg_bins[1:]) / 2
            tracked_var_bin_centers = (fixed_var_bins[:-1] + fixed_var_bins[1:]) / 2

            avg_kdes: list[np.ndarray] = []
            var_kdes: list[np.ndarray] = []
            for _ in range(NUM_BOOTSTRAP_ITERATIONS):
                sampled_indices = rng.choice(
                    len(tracked_avg_all), size=num_trajectories_grid, replace=True
                )
                sample_avg = tracked_avg_all[sampled_indices]
                _, avg_kde_values = compute_kde_on_bins(
                    data=sample_avg,
                    bin_width=bin_width_averages,
                    kernel_name=kernel_names_dict[column_name],
                    kernel_bandwidth=1.5 * bin_width_averages,
                    kernel_period=period,
                    bins=fixed_avg_bins,
                )
                avg_kdes.append(avg_kde_values)
                sample_var = tracked_var_all[sampled_indices]
                _, var_kde_values = compute_kde_on_bins(
                    data=sample_var,
                    bin_width=bin_width_variances,
                    kernel_name="gaussian",
                    kernel_bandwidth=1.5 * bin_width_variances,
                    kernel_period=None,
                    bins=fixed_var_bins,
                )
                var_kdes.append(var_kde_values)

            avg_kdes_arr = np.array(avg_kdes)
            var_kdes_arr = np.array(var_kdes)
            bootstrap_tracked_avg_kde[column_name] = {
                "bin_centers": tracked_avg_bin_centers,
                "mean": np.nanmean(avg_kdes_arr, axis=0),
                "ci_lower": np.nanpercentile(avg_kdes_arr, FP_CI_LOWER_PERCENTILE, axis=0),
                "ci_upper": np.nanpercentile(avg_kdes_arr, FP_CI_UPPER_PERCENTILE, axis=0),
            }
            bootstrap_tracked_var_kde[column_name] = {
                "bin_centers": tracked_var_bin_centers,
                "mean": np.nanmean(var_kdes_arr, axis=0),
                "ci_lower": np.nanpercentile(var_kdes_arr, FP_CI_LOWER_PERCENTILE, axis=0),
                "ci_upper": np.nanpercentile(var_kdes_arr, FP_CI_UPPER_PERCENTILE, axis=0),
            }

        # plot histograms of the column averages and variances across
        # trajectories for each column and crop pattern, with KDE overlaid.
        # Spline smoothing from the coarse bin-center grid to the fine x_eval
        # grid is applied here, at plot time.
        for column_name in column_names:
            variable_label = variable_labels_dict[column_name]
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            x_eval_avg_grid = x_eval_avg_dict["grid"][column_name]
            x_eval_avg_tracked = x_eval_avg_dict["tracked"][column_name]
            x_eval_var_grid = x_eval_var_dict["grid"][column_name]
            x_eval_var_tracked = x_eval_var_dict["tracked"][column_name]

            # --- grid: spline-smooth the native-grid KDE to x_eval, then plot ---
            grid_avg_smooth = smooth_kde_with_spline(
                bin_centers=grid_avg_kde_dict[column_name]["bin_centers"],
                kde_values=grid_avg_kde_dict[column_name]["kde_values"],
                x_eval=x_eval_avg_grid,
            )
            axes_title_avg = f"Histogram of average {variable_label} across trajectories"
            axes_xlabel_avg = f"$\\langle${variable_label}$\\rangle$"
            axes_ylabel_avg = f"P({axes_xlabel_avg})"
            plot_kde(
                axes=ax[0],
                x_eval=x_eval_avg_grid,
                kde_values=grid_avg_smooth,
                kde_line_style="-",
                kde_label="grid",
                axes_title=axes_title_avg,
                axes_xlabel=axes_xlabel_avg,
                axes_ylabel=axes_ylabel_avg,
                axes_xlimits=bin_limits_dict[column_name],
            )
            grid_var_smooth = smooth_kde_with_spline(
                bin_centers=grid_var_kde_dict[column_name]["bin_centers"],
                kde_values=grid_var_kde_dict[column_name]["kde_values"],
                x_eval=x_eval_var_grid,
            )
            axes_title_var = f"Histogram of variance {variable_label} across trajectories"
            axes_xlabel_var = f"Var({variable_label})"
            axes_ylabel_var = f"P({axes_xlabel_var})"
            plot_kde(
                axes=ax[1],
                x_eval=x_eval_var_grid,
                kde_values=grid_var_smooth,
                kde_line_style="-",
                kde_label="grid",
                axes_title=axes_title_var,
                axes_xlabel=axes_xlabel_var,
                axes_ylabel=axes_ylabel_var,
                axes_xlimits=(-0.01, 0.8),
            )

            # --- tracked: spline-smooth the bootstrap mean and CI from the native
            #     bin-center grid to x_eval, then plot ---
            tracked_avg_boot = bootstrap_tracked_avg_kde[column_name]
            tracked_var_boot = bootstrap_tracked_var_kde[column_name]
            tracked_avg_centers = tracked_avg_boot["bin_centers"]
            tracked_var_centers = tracked_var_boot["bin_centers"]

            tracked_avg_mean_smooth = smooth_kde_with_spline(
                tracked_avg_centers, tracked_avg_boot["mean"], x_eval_avg_tracked
            )
            tracked_avg_ci_lower_smooth = smooth_kde_with_spline(
                tracked_avg_centers, tracked_avg_boot["ci_lower"], x_eval_avg_tracked
            )
            tracked_avg_ci_upper_smooth = smooth_kde_with_spline(
                tracked_avg_centers, tracked_avg_boot["ci_upper"], x_eval_avg_tracked
            )
            (tracked_avg_line,) = ax[0].plot(
                x_eval_avg_tracked,
                tracked_avg_mean_smooth,
                linestyle="--",
                linewidth=2.0,
                label="tracked (bootstrap mean)",
            )
            ax[0].fill_between(
                x_eval_avg_tracked,
                tracked_avg_ci_lower_smooth,
                tracked_avg_ci_upper_smooth,
                color=tracked_avg_line.get_color(),
                alpha=0.25,
                label=f"tracked ({int(FP_CI_LOWER_PERCENTILE)}-{int(FP_CI_UPPER_PERCENTILE)}% CI)",
            )
            ax[0].legend(loc="upper right")

            tracked_var_mean_smooth = smooth_kde_with_spline(
                tracked_var_centers, tracked_var_boot["mean"], x_eval_var_tracked
            )
            tracked_var_ci_lower_smooth = smooth_kde_with_spline(
                tracked_var_centers, tracked_var_boot["ci_lower"], x_eval_var_tracked
            )
            tracked_var_ci_upper_smooth = smooth_kde_with_spline(
                tracked_var_centers, tracked_var_boot["ci_upper"], x_eval_var_tracked
            )
            (tracked_var_line,) = ax[1].plot(
                x_eval_var_tracked,
                tracked_var_mean_smooth,
                linestyle="--",
                linewidth=2.0,
                label="tracked (bootstrap mean)",
            )
            ax[1].fill_between(
                x_eval_var_tracked,
                tracked_var_ci_lower_smooth,
                tracked_var_ci_upper_smooth,
                color=tracked_var_line.get_color(),
                alpha=0.25,
                label=f"tracked ({int(FP_CI_LOWER_PERCENTILE)}-{int(FP_CI_UPPER_PERCENTILE)}% CI)",
            )
            ax[1].legend(loc="upper right")

            plt.suptitle(
                f"{plot_label}, grid vs. tracked crops \n "
                f"(grid n={num_trajectories_grid}, tracked n={num_trajectories_tracked}, "
                f"n={NUM_BOOTSTRAP_ITERATIONS} bootstrap samples, tracked n={num_trajectories_grid} per sample)"
            )
            plt.tight_layout()
            save_plot_to_path(
                fig,
                fig_savedir,
                f"{dataset_name_flow}_{column_name}",
            )

        if DEMO_MODE:
            logger.warning(
                "DEMO MODE: only processing one dataset for quick testing. Stopping after first dataset [ %s ].",
                dataset_name,
            )
            break


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
