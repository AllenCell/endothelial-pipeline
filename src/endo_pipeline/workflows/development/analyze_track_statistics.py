from endo_pipeline.cli import Datasets
from endo_pipeline.settings.track_statistics import (
    LONG_TRACK_THRESHOLD_LENGTH,
    NUM_TRACK_BOOTSTRAP_SAMPLES,
    TRACK_BOOTSRAP_CONFIDENCE_LEVEL,
)


def main(
    datasets: Datasets | None = None,
    min_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
    num_bootstrap: int = NUM_TRACK_BOOTSTRAP_SAMPLES,
    confidence_level: float = TRACK_BOOTSRAP_CONFIDENCE_LEVEL,
) -> None:
    import logging
    from typing import TypeAlias, cast

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
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
    from endo_pipeline.library.visualize.track_statistics import (
        plot_histogram_and_kde_with_confidence_interval,
        plot_histogram_with_kde,
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
    from endo_pipeline.settings.track_statistics import NUM_TRACK_BOOTSTRAP_SAMPLES_DEMO_MODE
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
        BaseResultsDict: TypeAlias = dict[ColumnName.DiffAEData, dict[str, np.ndarray]]
        hist_dict_grid: BaseResultsDict = {col: {} for col in column_names}
        kde_dict_grid: BaseResultsDict = {col: {} for col in column_names}
        hist_bins_dict: BaseResultsDict = {col: {} for col in column_names}
        hist_centers_dict: BaseResultsDict = {col: {} for col in column_names}
        kde_points_dict: BaseResultsDict = {col: {} for col in column_names}
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
                hist = np.nan_to_num(hist)
                kernel = KramersMoyalKernel(
                    name=kernel_name,
                    bandwidth=1.5 * hist_bin_width,
                    period=kernel_period,
                )
                hist_kde = get_kernel_density_estimate_from_histogram(
                    hist, bins=bins, kernel=kernel
                )
                hist_kde = np.nan_to_num(hist_kde)
                # interpolate between histogram centers for smoother KDE plot
                interp_centers = np.linspace(bins[0][0], bins[0][-1], 2000)
                spline = make_interp_spline(centers[0], hist_kde, k=3)  # k=3 for cubic spline
                hist_kde_smooth = spline(interp_centers)

                # add histogram and KDE to dict for grid data to compare with tracked data later
                hist_dict_grid[column_name][stat_name] = hist
                hist_bins_dict[column_name][stat_name] = bins[0]
                hist_centers_dict[column_name][stat_name] = centers[0]
                kde_dict_grid[column_name][stat_name] = hist_kde_smooth
                kde_points_dict[column_name][stat_name] = interp_centers

                # plot histogram of the column variance with KDE overlaid
                plot_histogram_with_kde(
                    ax[ax_index],
                    histogram=hist,
                    histogram_bins=bins[0],
                    histogram_kde=hist_kde_smooth,
                    kde_points=interp_centers,
                    histogram_color=hist_color,
                )
                ax[ax_index].set_title(
                    f"Histogram of {stat_name} {variable_label} across trajectories"
                )
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
            plt.close(fig)

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
        if num_trajectories_tracked <= num_trajectories_grid:
            logger.warning(
                "Dataset [ %s ] has more grid trajectories than tracked trajectories. "
                "Workflow does not support this case, skipping dataset.",
                dataset_name,
            )
            continue
        elif num_trajectories_grid < num_trajectories_tracked:
            logger.info(
                "Dataset [ %s ] has %d grid trajectories and %d tracked trajectories. "
                "Subsampling tracked trajectories to match number of grid trajectories for comparison.",
                dataset_name,
                num_trajectories_grid,
                num_trajectories_tracked,
            )
            BootstrapResultsDict: TypeAlias = dict[
                ColumnName.DiffAEData, dict[str, list[np.ndarray]]
            ]
            hist_tracked_list_dict: BootstrapResultsDict = {
                col: {stat: [] for stat in [column_average_str, column_variance_str]}
                for col in column_names
            }
            kde_tracked_list_dict: BootstrapResultsDict = {
                col: {stat: [] for stat in [column_average_str, column_variance_str]}
                for col in column_names
            }
            logger.debug(
                "Beginning bootstrap sampling of tracked trajectories for dataset [ %s ]",
                dataset_name,
            )
            if DEMO_MODE:
                num_bootstrap = NUM_TRACK_BOOTSTRAP_SAMPLES_DEMO_MODE
                logger.warning(
                    "DEMO MODE: reducing number of bootstrap samples to %d for quick testing.",
                    num_bootstrap,
                )
            for _ in range(num_bootstrap):
                sampled_traj_indices = rng.choice(
                    df_steady_state_tracked[ColumnName.CROP_INDEX].unique(),
                    size=num_trajectories_grid,
                    replace=False,
                )
                df_steady_state_subsampled = df_steady_state_tracked.copy()
                df_steady_state_subsampled = df_steady_state_subsampled[
                    df_steady_state_subsampled[ColumnName.CROP_INDEX].isin(sampled_traj_indices)
                ]
                df_with_stats_subsampled = compute_track_statistics(
                    df_steady_state_subsampled,
                    column_names,
                    trajectory_id_col=ColumnName.CROP_INDEX,
                    polar_angle_range=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE],
                    average_col_suffix=f"_{column_average_str}",
                    variance_col_suffix=f"_{column_variance_str}",
                )
                # compute histogram and KDE for each bootstrap sample and
                # average across samples for comparison with grid data
                for column_name in column_names:
                    # periodic kernel for polar angle average, non-periodic for variance
                    # (all variables) and average for non-polar angle variables
                    kernel_period_for_average = (
                        polar_angle_period
                        if column_name == ColumnName.DiffAEData.POLAR_ANGLE
                        else None
                    )

                    # loop over average and variance for the column to compute
                    # histograms and KDEs for both in the same loop
                    for (
                        stat_name,
                        hist_bin_width,
                        kernel_name,
                        kernel_period,
                        ax_index,
                        ax_xlim,
                    ) in zip(
                        [column_average_str, column_variance_str],
                        [bin_width_average, bin_width_variance],
                        [kernel_names_dict[column_name], kernel_name_for_variance],
                        [kernel_period_for_average, kernel_period_for_variance],
                        [0, 1],
                        [bin_limits_dict[column_name], x_limits_for_variance],
                        strict=True,
                    ):
                        data = df_with_stats_subsampled[f"{column_name}_{stat_name}"].to_numpy()

                        bins = hist_bins_dict[column_name][stat_name]
                        centers = hist_centers_dict[column_name][stat_name]
                        hist = np.histogram(data, bins=bins, density=True)[0]
                        hist = np.nan_to_num(hist)
                        kernel = KramersMoyalKernel(
                            name=kernel_name,
                            bandwidth=1.5 * hist_bin_width,
                            period=kernel_period,
                        )
                        hist_kde = get_kernel_density_estimate_from_histogram(
                            hist, bins=[bins], kernel=kernel
                        )
                        hist_kde = np.nan_to_num(hist_kde)
                        # interpolate between histogram centers for smoother KDE plot
                        kde_points = kde_points_dict[column_name][stat_name]
                        spline = make_interp_spline(centers, hist_kde, k=3)
                        hist_kde_smooth = spline(kde_points)

                        # add histogram and KDE to list for averaging across
                        # bootstrap samples for comparison with grid data later
                        hist_tracked_list_dict[column_name][stat_name].append(hist)
                        kde_tracked_list_dict[column_name][stat_name].append(hist_kde_smooth)
            logger.debug(
                "Completed bootstrap sampling of tracked trajectories for dataset [ %s ]",
                dataset_name,
            )
            # get mean and confidence intervals across bootstrap samples for histogram and KDE
            hist_tracked_means = {
                col: {
                    stat: np.mean(hist_tracked_list_dict[col][stat], axis=0)
                    for stat in hist_tracked_list_dict[col]
                }
                for col in column_names
            }
            hist_tracked_ci_lower = {
                col: {
                    stat: np.percentile(
                        hist_tracked_list_dict[col][stat], (1 - confidence_level) / 2 * 100, axis=0
                    )
                    for stat in hist_tracked_list_dict[col]
                }
                for col in column_names
            }
            hist_tracked_ci_upper = {
                col: {
                    stat: np.percentile(
                        hist_tracked_list_dict[col][stat], (1 + confidence_level) / 2 * 100, axis=0
                    )
                    for stat in hist_tracked_list_dict[col]
                }
                for col in column_names
            }
            kde_tracked_means = {
                col: {
                    stat: np.mean(kde_tracked_list_dict[col][stat], axis=0)
                    for stat in kde_tracked_list_dict[col]
                }
                for col in column_names
            }
            kde_tracked_ci_lower = {
                col: {
                    stat: np.percentile(
                        kde_tracked_list_dict[col][stat], (1 - confidence_level) / 2 * 100, axis=0
                    )
                    for stat in kde_tracked_list_dict[col]
                }
                for col in column_names
            }
            kde_tracked_ci_upper = {
                col: {
                    stat: np.percentile(
                        kde_tracked_list_dict[col][stat], (1 + confidence_level) / 2 * 100, axis=0
                    )
                    for stat in kde_tracked_list_dict[col]
                }
                for col in column_names
            }
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
                    histogram_bins = hist_bins_dict[column_name][stat_name]
                    kde_points = kde_points_dict[column_name][stat_name]

                    # plot histogram of the column variance with KDE overlaid
                    plot_histogram_and_kde_with_confidence_interval(
                        ax[ax_index],
                        histogram=hist_tracked_means[column_name][stat_name],
                        histogram_bins=histogram_bins,
                        histogram_confidence_interval=(
                            hist_tracked_ci_lower[column_name][stat_name],
                            hist_tracked_ci_upper[column_name][stat_name],
                        ),
                        histogram_kde=kde_tracked_means[column_name][stat_name],
                        kde_points=kde_points,
                        kde_confidence_interval=(
                            kde_tracked_ci_lower[column_name][stat_name],
                            kde_tracked_ci_upper[column_name][stat_name],
                        ),
                        histogram_color=hist_color,
                    )
                    ax[ax_index].set_title(
                        f"Histogram of {stat_name} {variable_label} across trajectories"
                    )
                    ax[ax_index].set_xlim(ax_xlim)
                    # plot labels: dynamically replace {{label}} in label wrapper with variable label
                    label_wrapper = axes_base_labels[stat_name]
                    ax[ax_index].set_xlabel(label_wrapper.replace("{{label}}", variable_label))
                    ax[ax_index].set_ylabel(
                        f"P({label_wrapper.replace('{{label}}', variable_label)})"
                    )
                plt.suptitle(
                    f"{plot_label}, tracked crops (subsampled {num_bootstrap} times to n={num_trajectories_grid} trajectories)"
                )
                plt.tight_layout()
                save_plot_to_path(
                    fig,
                    fig_savedir,
                    f"{dataset_name_flow}_{column_name}_statistics_histograms_tracked",
                )
                plt.close(fig)

                # finally, plot grid and tracked KDEs on the same plot for
                # direct comparison
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(
                    kde_points_dict[column_name][stat_name],
                    kde_dict_grid[column_name][stat_name],
                    color=hist_color,
                    linewidth=1.5,
                    linestyle="-",
                    label="grid",
                )
                ax.plot(
                    kde_points,
                    kde_tracked_means[column_name][stat_name],
                    color="k",
                    linewidth=1.5,
                    linestyle="--",
                    label="tracked",
                )
                ax.fill_between(
                    kde_points,
                    kde_tracked_ci_lower[column_name][stat_name],
                    kde_tracked_ci_upper[column_name][stat_name],
                    color="k",
                    alpha=0.3,
                    label=f"{int(confidence_level * 100)}% CI (tracked)",
                )
                ax.set_title(f"KDE of {stat_name} {variable_label} across trajectories")
                ax.set_xlim(ax_xlim)
                ax.set_xlabel(label_wrapper.replace("{{label}}", variable_label))
                ax.set_ylabel(f"P({label_wrapper.replace('{{label}}', variable_label)})")
                ax.legend()
                plt.suptitle(f"{plot_label}, grid vs tracked comparison")
                plt.tight_layout()
                save_plot_to_path(
                    fig,
                    fig_savedir,
                    f"{dataset_name_flow}_{column_name}_statistics_kde_comparison",
                )
                plt.close(fig)

        if DEMO_MODE:
            logger.warning(
                "DEMO MODE: only processing one dataset for quick testing. Stopping after first dataset [ %s ].",
                dataset_name,
            )
            break


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
