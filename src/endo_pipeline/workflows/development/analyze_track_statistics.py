from endo_pipeline.cli import Datasets
from endo_pipeline.settings.bootstrap_fixed_points import (
    FP_CI_LOWER_PERCENTILE,
    FP_CI_UPPER_PERCENTILE,
    NUM_BOOTSTRAP_ITERATIONS,
)
from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH


def main(
    datasets: Datasets | None = None,
    min_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
    n_bootstrap: int = NUM_BOOTSTRAP_ITERATIONS,
    ci_lower: float = FP_CI_LOWER_PERCENTILE,
    ci_upper: float = FP_CI_UPPER_PERCENTILE,
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
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.numerics.temporal_stats import (
        compute_kde_on_bins,
        process_dataframe_for_track_statistics,
    )
    from endo_pipeline.library.visualize.diffae_features.track_statistics import (
        plot_kde_for_track_statistics,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        POLAR_ANGLE_PERIOD,
    )
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
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
    columns_to_compute_grid = [*METADATA_COLUMNS_TO_KEEP["grid"], *column_names]
    columns_to_compute_tracked = [*METADATA_COLUMNS_TO_KEEP["tracked"], *column_names]

    # kernel names for KDEs
    kernel_names_dict = cast(dict[str | Column.DiffAEData, str], KERNEL_NAMES_DYNAMICS.copy())

    # bin widths for histograms of column averages and variances across
    # trajectories (currently hardcoded)
    bin_width_averages = 0.1
    bin_width_variances = 0.02
    ci_line_kwargs = {"alpha": 0.25, "label": f"{int(ci_lower)}-{int(ci_upper)}% CI"}

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

    # default bin limits (used for axes limits) and period for circular
    # variables (used for KDE computation)
    bin_limits_dict = BIN_LIMITS_DYNAMICS

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

        dataset_config = load_dataset_config(dataset_name)
        shear_stress = dataset_config.flow_conditions[0].shear_stress
        dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
        plot_label = f"{dataset_name} ({shear_stress} dyn/cm{Unicode.SQUARED})"
        fig_savedir = get_output_path(__file__, dataset_name)

        # load dataframe and perform additional filtering (e.g., remove
        # non-steady-state timepoints based on annotations), computing only the
        # columns needed for analysis
        df_grid_ = load_dataframe(
            grid_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_grid: pd.DataFrame = df_grid_[columns_to_compute_grid].compute()
        df_steady_state_grid = process_dataframe_for_track_statistics(
            df_grid, dataset_config, min_track_length
        )
        num_trajectories_grid = df_steady_state_grid[Column.CROP_INDEX].nunique()

        df_tracked_ = load_dataframe(
            tracked_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_tracked: pd.DataFrame = df_tracked_[columns_to_compute_tracked].compute()
        df_steady_state_tracked = process_dataframe_for_track_statistics(
            df_tracked, dataset_config, min_track_length
        )
        num_trajectories_tracked = df_steady_state_tracked[Column.CROP_INDEX].nunique()

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

        base_df = pd.DataFrame(columns=[Column.CROP_INDEX, *column_names])
        column_avg_df_dict: dict[str, pd.DataFrame] = {
            "grid": base_df.copy(),
            "tracked": base_df.copy(),
        }
        column_variance_df_dict: dict[str, pd.DataFrame] = {
            "grid": base_df.copy(),
            "tracked": base_df.copy(),
        }
        # Store bins and KDE evaluation points for each crop pattern and column
        # for later use in plotting and analysis
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

            # After computing the average and variance for each trajectory, drop
            # any remaining NaN values and use the resulting data to compute bin
            # edges for histograms and evaluation points for KDE for each column and
            # crop pattern. This ensures the bins and KDE evaluation points are
            # well-suited to the actual distribution of the data for each crop
            # pattern and column.
            for column_name in column_names:
                avg_data = (
                    column_avg_df_dict[crop_pattern][column_name].dropna().to_numpy().reshape(-1, 1)
                )
                avg_bins = get_bins(bin_widths=(bin_width_averages,), data=avg_data)[0]
                bins_avg_dict[crop_pattern][column_name] = avg_bins[0]
                x_eval_avg_dict[crop_pattern][column_name] = np.linspace(
                    avg_bins[0][0], avg_bins[0][-1], 2000
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
                    var_bins[0][0], var_bins[0][-1], 2000
                )

        # Compute histogram and KDE for each column and crop pattern, storing
        # the KDEs in a dictionary for later use in plotting and analysis.
        grid_avg_kde_dict: dict = {}
        grid_var_kde_dict: dict = {}
        tracked_avg_kde_dict: dict = {}
        tracked_var_kde_dict: dict = {}
        for column_name in column_names:
            period = POLAR_ANGLE_PERIOD if column_name == Column.DiffAEData.POLAR_ANGLE else None

            for crop_pattern, avg_kde_dict, var_kde_dict in [
                ("grid", grid_avg_kde_dict, grid_var_kde_dict),
                ("tracked", tracked_avg_kde_dict, tracked_var_kde_dict),
            ]:
                avg_data_all = column_avg_df_dict[crop_pattern][column_name].dropna().to_numpy()
                var_data_all = (
                    column_variance_df_dict[crop_pattern][column_name].dropna().to_numpy()
                )
                if crop_pattern == "grid":
                    avg_bin_centers, avg_kde_values = compute_kde_on_bins(
                        data=avg_data_all,
                        bin_width=bin_width_averages,
                        kernel_name=kernel_names_dict[column_name],
                        kernel_bandwidth=1.5 * bin_width_averages,
                        kernel_period=period,
                        bins=bins_avg_dict[crop_pattern][column_name],
                    )
                    avg_kde_dict[column_name] = {
                        "bin_centers": avg_bin_centers,
                        "kde_values": avg_kde_values,
                        "ci_lower": None,
                        "ci_upper": None,
                    }
                    var_bin_centers, var_kde_values = compute_kde_on_bins(
                        data=var_data_all,
                        bin_width=bin_width_variances,
                        kernel_name="gaussian",
                        kernel_bandwidth=1.5 * bin_width_variances,
                        kernel_period=None,
                        bins=bins_var_dict[crop_pattern][column_name],
                    )
                    var_kde_dict[column_name] = {
                        "bin_centers": var_bin_centers,
                        "kde_values": var_kde_values,
                        "ci_lower": None,
                        "ci_upper": None,
                    }
                elif crop_pattern == "tracked":
                    # use fixed bins derived from the full tracked data so all bootstrap
                    # samples share the same bin-center grid and can be stacked directly
                    fixed_avg_bins = bins_avg_dict[crop_pattern][column_name]
                    fixed_var_bins = bins_var_dict[crop_pattern][column_name]
                    tracked_avg_bin_centers = (fixed_avg_bins[:-1] + fixed_avg_bins[1:]) / 2
                    tracked_var_bin_centers = (fixed_var_bins[:-1] + fixed_var_bins[1:]) / 2

                    avg_kdes: list[np.ndarray] = []
                    var_kdes: list[np.ndarray] = []
                    # Begin bootstrap procedure
                    for _ in range(n_bootstrap):
                        # Sample trajectories with replacement from the tracked
                        # data, then compute KDEs for the average and variance
                        # of the column across trajectories for this bootstrap
                        # sample. Using the same fixed bins for each bootstrap
                        # sample allows us to directly compare the KDEs across
                        # bootstrap iterations and compute confidence intervals
                        # at each bin center.
                        sampled_indices = rng.choice(
                            len(avg_data_all), size=num_trajectories_grid, replace=True
                        )
                        sample_avg = avg_data_all[sampled_indices]
                        _, avg_kde_values = compute_kde_on_bins(
                            data=sample_avg,
                            bin_width=bin_width_averages,
                            kernel_name=kernel_names_dict[column_name],
                            kernel_bandwidth=1.5 * bin_width_averages,
                            kernel_period=period,
                            bins=fixed_avg_bins,
                        )
                        avg_kdes.append(avg_kde_values)
                        sample_var = var_data_all[sampled_indices]
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
                    avg_kde_dict[column_name] = {
                        "bin_centers": tracked_avg_bin_centers,
                        "kde_values": np.nanmean(avg_kdes_arr, axis=0),
                        "ci_lower": np.nanpercentile(avg_kdes_arr, ci_lower, axis=0),
                        "ci_upper": np.nanpercentile(avg_kdes_arr, ci_upper, axis=0),
                    }
                    var_kde_dict[column_name] = {
                        "bin_centers": tracked_var_bin_centers,
                        "kde_values": np.nanmean(var_kdes_arr, axis=0),
                        "ci_lower": np.nanpercentile(var_kdes_arr, ci_lower, axis=0),
                        "ci_upper": np.nanpercentile(var_kdes_arr, ci_upper, axis=0),
                    }
        for column_name in column_names:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            for crop_pattern, kde_avg_dict, kde_var_dict in [
                ("grid", grid_avg_kde_dict, grid_var_kde_dict),
                ("tracked", tracked_avg_kde_dict, tracked_var_kde_dict),
            ]:
                kde_line_kwargs = {
                    "color": "k",
                    "linewidth": 2,
                    "linestyle": "-" if crop_pattern == "grid" else "--",
                    "label": (
                        crop_pattern
                        if crop_pattern == "grid"
                        else f"{crop_pattern} (bootstrap mean)"
                    ),
                }
                plot_kde_for_track_statistics(
                    ax=ax[0],
                    kde_values=kde_avg_dict[column_name]["kde_values"],
                    bin_centers=kde_avg_dict[column_name]["bin_centers"],
                    x_eval=x_eval_avg_dict[crop_pattern][column_name],
                    kde_ci_lower=kde_avg_dict[column_name]["ci_lower"],
                    kde_ci_upper=kde_avg_dict[column_name]["ci_upper"],
                    axes_title=f"{column_name} average KDE - {crop_pattern}",
                    axes_xlabel=f"{column_name} average",
                    axes_ylabel="KDE",
                    kde_line_kwargs=kde_line_kwargs,
                    ci_line_kwargs=ci_line_kwargs,
                )
                plot_kde_for_track_statistics(
                    ax=ax[1],
                    kde_values=kde_var_dict[column_name]["kde_values"],
                    bin_centers=kde_var_dict[column_name]["bin_centers"],
                    x_eval=x_eval_var_dict[crop_pattern][column_name],
                    kde_ci_lower=kde_var_dict[column_name]["ci_lower"],
                    kde_ci_upper=kde_var_dict[column_name]["ci_upper"],
                    axes_title=f"{column_name} variance KDE - {crop_pattern}",
                    axes_xlabel=f"{column_name} variance",
                    axes_ylabel="KDE",
                    kde_line_kwargs=kde_line_kwargs,
                    ci_line_kwargs=ci_line_kwargs,
                )
            plt.suptitle(
                f"{plot_label}, grid vs. tracked crops \n "
                f"(grid n={num_trajectories_grid}, tracked n={num_trajectories_tracked}, "
                f"n={n_bootstrap} bootstrap samples, tracked n={num_trajectories_grid} per sample)"
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
