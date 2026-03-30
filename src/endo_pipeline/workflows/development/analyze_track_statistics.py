from endo_pipeline.cli import Datasets
from endo_pipeline.settings.track_statistics import LONG_TRACK_THRESHOLD_LENGTH


def main(
    datasets: Datasets | None = None,
    min_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
) -> None:
    import logging
    from typing import cast

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import to_rgb
    from scipy.interpolate import make_interp_spline

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        filter_dataframe_by_annotations,
        filter_dataframe_by_track_length,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate_from_histogram,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.track_statistics import compute_track_statistics
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_dataset_color,
        get_label_for_column,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        RESCALE_THETA,
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
    column_names: list[ColumnName.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)
    variable_labels_dict = {
        col: get_label_for_column(col).replace("polar ", "") for col in column_names
    }
    columns_to_compute_grid = [*METADATA_COLUMNS_TO_KEEP["grid"], *column_names]
    columns_to_compute_tracked = [*METADATA_COLUMNS_TO_KEEP["tracked"], *column_names]
    kernel_names_dict = cast(dict[str | ColumnName.DiffAEData, str], KERNEL_NAMES_DYNAMICS.copy())

    # settings for histogram plotting of trajectory statistics
    column_average_str = "average"
    column_variance_str = "variance"
    bin_width_average = 0.1
    bin_width_variance = 0.02
    axes_base_labels = {
        column_average_str: "$\\langle${{label}}$\\rangle$",
        column_variance_str: "Var({{label}})",
    }
    # use Gaussian kernel for variance histograms regardless of variable since
    # variance is non-periodic
    kernel_name_for_variance = "gaussian"
    kernel_period_for_variance = None
    x_limits_for_variance = (-0.01, 0.8)

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

    # unpack default bin widths and limits for each column, adjusting limits if rescaling theta
    bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
    polar_angle_period = (
        bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1]
        - bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0]
    )

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

        hist_color = get_dataset_color(dataset_name)
        shear_stress = dataset_config.flow_conditions[0].shear_stress
        dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
        plot_label = f"{dataset_name} ({shear_stress} dyn/cm$^2$)"
        fig_savedir = get_output_path(__file__, dataset_name)

        # First: get stats and histograms for grid based trajectories, which are
        # not affected by track length filtering, to determine how many tracked
        # trajectories to sample for comparison.
        df_grid_ = load_dataframe(
            grid_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_grid: pd.DataFrame = df_grid_[columns_to_compute_grid].compute()
        df_steady_state_grid = filter_dataframe_by_annotations(
            df_grid,
            load_dataset_config(dataset_name),
            timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
        )
        num_trajectories_grid = df_steady_state_grid[ColumnName.CROP_INDEX].nunique()
        df_with_stats_grid = compute_track_statistics(
            df_steady_state_grid.copy(),
            column_names,
            trajectory_id_col=ColumnName.CROP_INDEX,
            polar_angle_range=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE],
            average_col_suffix=f"_{column_average_str}",
            variance_col_suffix=f"_{column_variance_str}",
        )
        # get histogram of the column average using bin widths of 0.1,
        # adjusting x-axis limits based on bin limits for the column
        hist_dict_grid: dict[ColumnName.DiffAEData, dict[str, np.ndarray]] = {
            column_name: {} for column_name in column_names
        }
        kde_dict_grid: dict[ColumnName.DiffAEData, dict[str, np.ndarray]] = {
            column_name: {} for column_name in column_names
        }
        hist_points_dict: dict[ColumnName.DiffAEData, np.ndarray] = {}
        kde_points_dict: dict[ColumnName.DiffAEData, np.ndarray] = {}
        for column_name in column_names:
            # init plot and plot labels for the column
            variable_label = variable_labels_dict[column_name]
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # periodic kernel for polar angle average, non-periodic for variance
            # (all variables) and average for non-polar angle variables
            kernel_period_for_average = (
                polar_angle_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
            )

            # loop over average and variance for the column to compute
            # histograms and KDEs for both in the same loop
            for stat_name, hist_bin_width, kernel_name, kernel_period, ax_index, ax_xlim in zip(
                [column_average_str, column_variance_str],
                [bin_width_average, bin_width_variance],
                [kernel_names_dict[column_name], kernel_name_for_variance],
                [kernel_period_for_average, kernel_period_for_variance],
                [0, 1],
                [bin_limits_dict[column_name], x_limits_for_variance],
                strict=True,
            ):
                data = df_with_stats_grid[f"{column_name}_{stat_name}"].to_numpy()

                bins, centers = get_bins(bin_widths=(hist_bin_width,), data=data, pad=0)
                hist = np.histogram(data, bins=bins[0], density=True)[0]
                kernel = KramersMoyalKernel(
                    name=kernel_name,
                    bandwidth=1.5 * hist_bin_width,
                    period=kernel_period,
                )
                hist_kde = get_kernel_density_estimate_from_histogram(
                    hist, bins=bins, kernel=kernel
                )
                # interpolate between histogram centers for smoother KDE plot
                interp_centers = np.linspace(bins[0][0], bins[0][-1], 2000)
                spline = make_interp_spline(centers[0], hist_kde, k=3)  # k=3 for cubic spline
                hist_kde_smooth = spline(interp_centers)

                # add histogram and KDE to dict for grid data to compare with tracked data later
                hist_dict_grid[column_name][stat_name] = hist
                hist_points_dict[column_name] = bins[0]
                kde_dict_grid[column_name][stat_name] = hist_kde_smooth
                kde_points_dict[column_name] = interp_centers

                # plot histogram of the column variance with KDE overlaid
                ax[ax_index].bar(
                    bins[0][:-1],
                    hist,
                    width=np.diff(bins[0]),
                    color=(*to_rgb(hist_color), 0.5),
                    edgecolor=(*to_rgb("k"), 1.0),
                    align="edge",
                )
                ax[ax_index].plot(
                    interp_centers,
                    hist_kde_smooth,
                    color=hist_color,
                    linewidth=1.5,
                )
                ax[ax_index].set_title(f"Histogram of average {variable_label} across trajectories")
                ax[ax_index].set_xlim(ax_xlim)
                # plot labels: dynamically replace {{label}} in label wrapper with variable label
                label_wrapper = axes_base_labels[stat_name]
                ax[ax_index].set_xlabel(label_wrapper.replace("{{label}}", variable_label))
                ax[ax_index].set_ylabel(f"P({label_wrapper.replace('{{label}}', variable_label)})")
            plt.suptitle(f"{plot_label}, tracked crops (n={num_trajectories_grid} trajectories)")
            plt.tight_layout()
            save_plot_to_path(
                fig,
                fig_savedir,
                f"{dataset_name_flow}_{column_name}_statistics_histograms_grid",
            )

        # Next: get stats and histograms for tracked trajectories, filtering by
        # track length and subsampling to match number of grid trajectories for
        # comparison if specified.
        df_tracked_ = load_dataframe(
            tracked_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_tracked: pd.DataFrame = df_tracked_[columns_to_compute_tracked].compute()
        df_steady_state_tracked = filter_dataframe_by_annotations(
            df_tracked,
            load_dataset_config(dataset_name),
            timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
        )
        # Perform additional filtering by track length
        df_steady_state_tracked = filter_dataframe_by_track_length(
            df_steady_state_tracked, ColumnName.TRACK_LENGTH, minimum_track_length=min_track_length
        )
        num_trajectories_tracked = df_steady_state_tracked[ColumnName.CROP_INDEX].nunique()

        # subsample trajectories if num_subsample is specified and there are
        # more than num_subsample trajectories
        if num_trajectories_grid < num_trajectories_tracked:
            logger.info(
                "Dataset [ %s ] has %d grid trajectories and %d tracked trajectories. "
                "Subsampling tracked trajectories to match number of grid trajectories for comparison.",
                dataset_name,
                num_trajectories_grid,
                num_trajectories_tracked,
            )
            sampled_traj_indices = rng.choice(
                df_steady_state_tracked[ColumnName.CROP_INDEX].unique(),
                size=num_trajectories_grid,
                replace=False,
            )
            df_steady_state_tracked = df_steady_state_tracked[
                df_steady_state_tracked[ColumnName.CROP_INDEX].isin(sampled_traj_indices)
            ]
        elif num_trajectories_tracked < num_trajectories_grid:
            logger.warning(
                "Dataset [ %s ] has more grid trajectories than tracked trajectories. "
                "Not subsampling tracked trajectories, but this may affect comparison between grid and tracked statistics.",
                dataset_name,
            )
        df_with_stats_tracked = compute_track_statistics(
            df_steady_state_tracked.copy(),
            column_names,
            trajectory_id_col=ColumnName.CROP_INDEX,
            polar_angle_range=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE],
            average_col_suffix=f"_{column_average_str}",
            variance_col_suffix=f"_{column_variance_str}",
        )

        # make histograms and KDEs for tracked data
        for column_name in column_names:
            # init plot and plot labels for the column
            variable_label = variable_labels_dict[column_name]
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # periodic kernel for polar angle average, non-periodic for variance
            # (all variables) and average for non-polar angle variables
            kernel_period_for_average = (
                polar_angle_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
            )

            # loop over average and variance for the column to compute
            # histograms and KDEs for both in the same loop
            for stat_name, hist_bin_width, kernel_name, kernel_period, ax_index, ax_xlim in zip(
                [column_average_str, column_variance_str],
                [bin_width_average, bin_width_variance],
                [kernel_names_dict[column_name], kernel_name_for_variance],
                [kernel_period_for_average, kernel_period_for_variance],
                [0, 1],
                [bin_limits_dict[column_name], x_limits_for_variance],
                strict=True,
            ):
                data = df_with_stats_tracked[f"{column_name}_{stat_name}"].to_numpy()

                bins, centers = get_bins(bin_widths=(hist_bin_width,), data=data, pad=0)
                hist = np.histogram(data, bins=bins[0], density=True)[0]
                kernel = KramersMoyalKernel(
                    name=kernel_name,
                    bandwidth=1.5 * hist_bin_width,
                    period=kernel_period,
                )
                hist_kde = get_kernel_density_estimate_from_histogram(
                    hist, bins=bins, kernel=kernel
                )
                # interpolate between histogram centers for smoother KDE plot
                interp_centers = np.linspace(bins[0][0], bins[0][-1], 2000)
                spline = make_interp_spline(centers[0], hist_kde, k=3)  # k=3 for cubic spline
                hist_kde_smooth = spline(interp_centers)

                # plot histogram of the column variance with KDE overlaid
                ax[ax_index].bar(
                    bins[0][:-1],
                    hist,
                    width=np.diff(bins[0]),
                    color=(*to_rgb(hist_color), 0.5),
                    edgecolor=(*to_rgb("k"), 1.0),
                    align="edge",
                )
                ax[ax_index].plot(
                    interp_centers,
                    hist_kde_smooth,
                    color=hist_color,
                    linewidth=1.5,
                )
                ax[ax_index].set_title(f"Histogram of average {variable_label} across trajectories")
                ax[ax_index].set_xlim(ax_xlim)
                # plot labels: dynamically replace {{label}} in label wrapper with variable label
                label_wrapper = axes_base_labels[stat_name]
                ax[ax_index].set_xlabel(label_wrapper.replace("{{label}}", variable_label))
                ax[ax_index].set_ylabel(f"P({label_wrapper.replace('{{label}}', variable_label)})")
            plt.suptitle(f"{plot_label}, tracked crops (n={num_trajectories_tracked} trajectories)")
            plt.tight_layout()
            save_plot_to_path(
                fig,
                fig_savedir,
                f"{dataset_name_flow}_{column_name}_statistics_histograms_tracked",
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
