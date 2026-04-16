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
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
    from endo_pipeline.library.visualize.features import get_label_for_column
    from endo_pipeline.library.visualize.track_statistics import plot_histogram_and_kde
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
    variable_labels_dict = {col: get_label_for_column(col) for col in column_names}
    columns_to_compute_grid = [*METADATA_COLUMNS_TO_KEEP["grid"], *column_names]
    columns_to_compute_tracked = [*METADATA_COLUMNS_TO_KEEP["tracked"], *column_names]

    # kernel names for KDEs
    kernel_names_dict = cast(dict[str | ColumnName.DiffAEData, str], KERNEL_NAMES_DYNAMICS.copy())

    # bin widths for histograms of column averages and variances across
    # trajectories (currently hardcoded)
    bin_width_averages = 0.1
    bin_width_variances = 0.02

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

        # load dataframe and perform additional filtering (e.g., remove
        # non-steady-state timepoints based on annotations), computing only the
        # columns needed for analysis
        dataset_config = load_dataset_config(dataset_name)
        df_grid_ = load_dataframe(
            grid_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_grid: pd.DataFrame = df_grid_[columns_to_compute_grid].compute()
        df_steady_state_grid = filter_dataframe_to_steady_state(df_grid, dataset_config)
        num_trajectories_grid = df_steady_state_grid[ColumnName.CROP_INDEX].nunique()

        df_tracked_ = load_dataframe(
            tracked_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_tracked: pd.DataFrame = df_tracked_[columns_to_compute_tracked].compute()
        df_steady_state_tracked = filter_dataframe_to_steady_state(df_tracked, dataset_config)
        # Perform additional filtering by track length
        df_steady_state_tracked = filter_dataframe_by_track_length(
            df_steady_state_tracked, min_track_length
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
        # trajectories for each column and crop pattern, with KDE overlaid
        for column_name in column_names:
            variable_label = variable_labels_dict[column_name]
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            period = (
                polar_angle_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
            )
            for crop_pattern, num_traj, line_style in [
                ("grid", num_trajectories_grid, "-"),
                ("tracked", num_trajectories_tracked, "--"),
            ]:
                # histogram and KDE for column average
                axes_title = f"Histogram of average {variable_label} across trajectories"
                axes_xlabel = f"$\\langle${variable_label}$\\rangle$"
                axes_ylabel = f"P({axes_xlabel})"
                plot_histogram_and_kde(
                    axes=ax[0],
                    data=column_avg_df_dict[crop_pattern][column_name].to_numpy(),
                    bin_width=bin_width_averages,
                    kernel_name=kernel_names_dict[column_name],
                    kernel_bandwidth=1.5 * bin_width_averages,
                    kernel_period=period,
                    hist_color=hist_color,
                    kde_line_style=line_style,
                    kde_label=crop_pattern,
                    axes_title=axes_title,
                    axes_xlabel=axes_xlabel,
                    axes_ylabel=axes_ylabel,
                    axes_xlimits=bin_limits_dict[column_name],
                )

                # histogram and KDE for column variance
                axes_title = f"Histogram of variance {variable_label} across trajectories"
                axes_xlabel = f"$\\mathrm{{Var}}({variable_label})$"
                axes_ylabel = f"P({axes_xlabel})"
                plot_histogram_and_kde(
                    axes=ax[1],
                    data=column_variance_df_dict[crop_pattern][column_name].to_numpy(),
                    bin_width=bin_width_variances,
                    kernel_name="gaussian",
                    kernel_bandwidth=1.5 * bin_width_variances,
                    kernel_period=None,
                    hist_color=hist_color,
                    kde_line_style=line_style,
                    kde_label=crop_pattern,
                    axes_title=axes_title,
                    axes_xlabel=axes_xlabel,
                    axes_ylabel=axes_ylabel,
                    axes_xlimits=(-0.01, 0.8),
                )

            plt.suptitle(f"{plot_label}, grid vs. tracked crops (n={num_traj} trajectories)")
            plt.tight_layout()
            save_plot_to_path(
                fig,
                fig_savedir,
                f"{dataset_name_flow}_{column_name}_statistics_histograms",
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
