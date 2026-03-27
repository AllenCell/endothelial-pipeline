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
    from matplotlib.colors import to_rgb
    from scipy.interpolate import make_interp_spline
    from scipy.stats import circmean, circvar

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

        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for analysis
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
        if num_trajectories_grid > num_trajectories_tracked:
            logger.info(
                "Dataset [ %s ] has %d grid trajectories and %d tracked trajectories. Subsampling grid trajectories to match number of tracked trajectories for comparison.",
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
        elif num_trajectories_tracked > num_trajectories_grid:
            logger.warning(
                "Dataset [ %s ] has more tracked trajectories than grid trajectories. Not subsampling tracked trajectories, but this may affect comparison between grid and tracked statistics.",
                dataset_name,
            )

        # put together grid and tracked dataframes for easier processing, adding
        # a column to indicate crop pattern
        df_steady_state_dict: dict[str, pd.DataFrame] = {
            "grid": df_steady_state_grid,
            "tracked": df_steady_state_tracked,
        }

        base_df = pd.DataFrame(columns=[ColumnName.CROP_INDEX, *column_names])
        column_avg_df_dict: dict[str, pd.DataFrame] = {
            "grid": base_df.copy(),
            "tracked": base_df.copy(),
        }
        column_variance_df_dict: dict[str, pd.DataFrame] = {
            "grid": base_df.copy(),
            "tracked": base_df.copy(),
        }
        for crop_pattern in ["grid", "tracked"]:
            for traj_index, df_traj in df_steady_state_dict[crop_pattern].groupby(
                ColumnName.CROP_INDEX
            ):
                for column_name in column_names:
                    if column_name == ColumnName.DiffAEData.POLAR_ANGLE:
                        # take circular mean for polar angle to account for periodicity
                        column_avg_df_dict[crop_pattern].loc[traj_index, column_name] = circmean(
                            df_traj[column_name],
                            high=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1],
                            low=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0],
                        )
                        column_variance_df_dict[crop_pattern].loc[traj_index, column_name] = (
                            circvar(
                                df_traj[column_name],
                                high=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1],
                                low=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0],
                            )
                        )
                    else:
                        column_avg_df_dict[crop_pattern].loc[traj_index, column_name] = np.nanmean(
                            df_traj[column_name]
                        )
                        column_variance_df_dict[crop_pattern].loc[traj_index, column_name] = (
                            np.nanvar(df_traj[column_name])
                        )

        # plot histograms of the column averages and variances across
        # trajectories for each column
        axes_base_label = ["$\\langle${{label}}$\\rangle$", "Var({{label}})"]
        histogram_bin_widths = [0.1, 0.02]
        for column_name in column_names:
            variable_label = variable_labels_dict[column_name]
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            period = (
                polar_angle_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
            )
            # plot histograms for grid and tracked data overlaid for comparison
            for crop_pattern, num_traj, line_style in zip(
                ["grid", "tracked"],
                [num_trajectories_grid, num_trajectories_tracked],
                ["-", "--"],
                strict=True,
            ):
                # loop over axes and other associated args for plotting average
                # and variance histograms in the same loop
                data_list: list[pd.DataFrame] = [
                    column_avg_df_dict[crop_pattern][column_name],
                    column_variance_df_dict[crop_pattern][column_name],
                ]
                for (
                    ax_index,
                    data,
                    label_wrapper,
                    bin_width,
                    kernel_name,
                    kernel_period,
                    axes_xlim,
                ) in zip(
                    [0, 1],
                    data_list,
                    axes_base_label,
                    histogram_bin_widths,
                    [kernel_names_dict[column_name], "gaussian"],
                    [period, None],
                    [bin_limits_dict[column_name], (-0.01, 0.8)],
                    strict=True,
                ):
                    # get histogram of the column average using bin widths of 0.1,
                    # adjusting x-axis limits based on bin limits for the column
                    bins, centers = get_bins(bin_widths=(bin_width,), data=data.to_numpy(), pad=0)
                    hist = np.histogram(data, bins=bins[0], density=True)[0]
                    kernel = KramersMoyalKernel(
                        name=kernel_name,
                        bandwidth=1.5 * bin_width,
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
                        linestyle=line_style,
                    )
                    ax[ax_index].set_title(
                        f"Histogram of average {variable_label} across trajectories"
                    )
                    ax[ax_index].set_xlim(axes_xlim)
                    # plot labels: dynamically replace {{label}} in label wrapper with variable label
                    ax[ax_index].set_xlabel(label_wrapper.replace("{{label}}", variable_label))
                    ax[ax_index].set_ylabel(
                        f"P({label_wrapper.replace('{{label}}', variable_label)})"
                    )

            plt.suptitle(f"{plot_label}, {crop_pattern} crops (n={num_traj} trajectories)")
            plt.tight_layout()
            save_plot_to_path(
                fig,
                fig_savedir,
                f"{dataset_name_flow}_{column_name}_statistics_histograms_{crop_pattern}",
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
