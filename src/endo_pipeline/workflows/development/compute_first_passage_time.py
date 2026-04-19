"""This workflow computes the time of first passage for each track in the dataset."""

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    minimum_track_length: int | None = None,
    run_FPT_threshold_parameter_sweep: bool = False,
) -> None:

    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs.dataset_config_io import (
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.integration.track_integration import (
        add_distance_to_fixed_points_columns,
        add_first_passage_time_column,
        compute_first_passage_time_parameter_sweep_df,
        compute_first_passage_time_stats_for_bins,
        load_filtered_trajectory_df_for_first_passage_time_workflow,
    )
    from endo_pipeline.library.analyze.numerics.binning import adjust_limits_from_bin_size, get_bins
    from endo_pipeline.library.analyze.numerics.fixed_points import (
        load_fixed_points_dataframe_for_dataset,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_first_passage_time_3d_scatter,
        plot_first_passage_time_correlation,
        plot_first_passage_time_parameter_sweep,
    )
    from endo_pipeline.settings import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.flow_field_3d import (
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        LONG_TRACK_THRESHOLD_LENGTH,
    )
    from endo_pipeline.settings.flow_field_dataframes import STABILITY_COLUMN_NAME
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE

    logger = logging.getLogger(__name__)

    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)

    if minimum_track_length is None:
        minimum_track_length = LONG_TRACK_THRESHOLD_LENGTH

    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.info(f"Running in demo mode, processing only the first dataset: {dataset_names}")

    for dataset_name in dataset_names:

        dataset_config = load_dataset_config(dataset_name)

        if DEMO_MODE:
            out_dir = get_output_path(__file__, "demo", dataset_name)
        else:
            out_dir = get_output_path(__file__, dataset_name)

        # load the dynamics features from the grid-based and track-based dataframes
        traj_df_grid = load_filtered_trajectory_df_for_first_passage_time_workflow(
            dataset_name,
            crop_pattern="grid",
            minimum_track_length=minimum_track_length,
        )
        traj_df_tracked = load_filtered_trajectory_df_for_first_passage_time_workflow(
            dataset_name,
            crop_pattern="tracked",
            minimum_track_length=minimum_track_length,
        )

        # load the flow field dictionaries and fixed points
        fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)

        if fixed_points_df.empty:
            logger.warning(f"No fixed points found for dataset {dataset_name}, skipping dataset.")
            continue

        # add the distances from the fixed points for the grid-based trajectories
        traj_df_grid = add_distance_to_fixed_points_columns(
            trajectory_df=traj_df_grid,
            fixed_point_df=fixed_points_df,
            trajectory_columns=DYNAMICS_COLUMN_NAMES,
        )

        # add the distances from the fixed points for the track-based trajectories
        traj_df_tracked = add_distance_to_fixed_points_columns(
            trajectory_df=traj_df_tracked,
            fixed_point_df=fixed_points_df,
            trajectory_columns=DYNAMICS_COLUMN_NAMES,
        )

        # 1. bin (theta, r, rho) feature space
        # define the bin sizes for each feature to be binned
        bin_sizes = {
            Column.DiffAEData.POLAR_ANGLE: np.deg2rad(15),
            Column.DiffAEData.POLAR_RADIUS: 0.25,
            Column.DiffAEData.PC3_FLIPPED: 0.5,
        }

        # get the data limits for each feature to be binned
        bin_limits: dict = {}
        for col in DYNAMICS_COLUMN_NAMES:
            col_min = min(traj_df_grid[col].min(), traj_df_tracked[col].min())
            col_max = max(traj_df_grid[col].max(), traj_df_tracked[col].max())
            bin_limits[col] = (col_min, col_max)

        # adjust the bin_limits if the feature has a defined range (e.g. for angles)
        defined_bin_limits = {
            Column.DiffAEData.POLAR_ANGLE: (0, np.pi),
            Column.DiffAEData.POLAR_RADIUS: (0, None),
            Column.DiffAEData.PC3_FLIPPED: (None, None),
        }
        for col in DYNAMICS_COLUMN_NAMES:
            if col in defined_bin_limits:
                bin_limits[col] = adjust_limits_from_bin_size(
                    data_min_max=bin_limits[col],
                    defined_min_max=defined_bin_limits[col],
                    bin_size=bin_sizes[col],
                )

        bin_widths = [bin_sizes[col] for col in DYNAMICS_COLUMN_NAMES]
        bin_limits = [bin_limits[col] for col in DYNAMICS_COLUMN_NAMES]
        bin_edges, bin_centers = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)

        # 2. identify trajectories that pass a fixed point and filter df to only those trajectories
        # find if and when a trajectory reaches a fixed point
        for fp_idx in fixed_points_df.index:
            # for now we will only look at first passage times to stable fixed points
            fp_stability = fixed_points_df.loc[fp_idx, STABILITY_COLUMN_NAME]
            if fp_stability != "stable":
                logger.info(
                    f"Fixed point {fp_idx} in dataset {dataset_name} is not stable (stability = "
                    f"{fp_stability}), skipping for first passage time analysis."
                )
                continue

            if run_FPT_threshold_parameter_sweep:
                # run a parameter sweep of the first passage times using different
                # thresholds for what it means to have "reached" the fixed point
                thresholds = np.linspace(0, 1, 41)
                traj_df_grid_param_sweep = traj_df_grid.copy()
                traj_df_tracked_param_sweep = traj_df_tracked.copy()
                traj_df_grid_param_sweep = compute_first_passage_time_parameter_sweep_df(
                    fixed_point_index=fp_idx,
                    trajectory_df=traj_df_grid_param_sweep,
                    thresholds=thresholds,
                )
                traj_df_tracked_param_sweep = compute_first_passage_time_parameter_sweep_df(
                    fixed_point_index=fp_idx,
                    trajectory_df=traj_df_tracked_param_sweep,
                    thresholds=thresholds,
                )

                # plot the parameter sweep results
                plot_first_passage_time_parameter_sweep(
                    dataset_config=dataset_config,
                    fixed_point_index=fp_idx,
                    fixed_point_stability=fp_stability,
                    first_passage_time_param_sweep_df=traj_df_grid_param_sweep,
                    out_dir=out_dir,
                )
                plot_first_passage_time_parameter_sweep(
                    dataset_config=dataset_config,
                    fixed_point_index=fp_idx,
                    fixed_point_stability=fp_stability,
                    first_passage_time_param_sweep_df=traj_df_tracked_param_sweep,
                    out_dir=out_dir,
                )

            traj_df_grid[f"is_at_fp_{fp_idx}"] = (
                traj_df_grid[f"dist_from_fp_{fp_idx}"] <= MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
            )
            traj_df_tracked[f"is_at_fp_{fp_idx}"] = (
                traj_df_tracked[f"dist_from_fp_{fp_idx}"] <= MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
            )

            traj_df_grid[f"traj_reached_fp_{fp_idx}"] = traj_df_grid.groupby(Column.CROP_INDEX)[
                f"is_at_fp_{fp_idx}"
            ].transform(any)
            traj_df_tracked[f"traj_reached_fp_{fp_idx}"] = traj_df_tracked.groupby(
                Column.CROP_INDEX
            )[f"is_at_fp_{fp_idx}"].transform(any)

            traj_df_grid_sub = traj_df_grid[traj_df_grid[f"traj_reached_fp_{fp_idx}"]]
            traj_df_tracked_sub = traj_df_tracked[traj_df_tracked[f"traj_reached_fp_{fp_idx}"]]

            # compute the timepoint at which each trajectory first reaches a fixed point
            traj_df_grid_sub = add_first_passage_time_column(
                fixed_point_index=fp_idx,
                trajectory_df=traj_df_grid_sub,
                column=f"dist_from_fp_{fp_idx}",
                threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
            )
            traj_df_tracked_sub = add_first_passage_time_column(
                fixed_point_index=fp_idx,
                trajectory_df=traj_df_tracked_sub,
                column=f"dist_from_fp_{fp_idx}",
                threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
            )

        # 3. for each bin (across all steady-state timepoints), compute the mean,
        #    median, and standard deviation of first-passage times for the trajectories
        time_to_first_passage_col_name = f"time_to_fp_{fp_idx}"

        fpt_stats_df_grid = compute_first_passage_time_stats_for_bins(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            trajectory_df=traj_df_grid_sub,
            time_to_first_passage_col_name=time_to_first_passage_col_name,
            feature_column_names=list(DYNAMICS_COLUMN_NAMES),
        )
        fpt_stats_df_tracked = compute_first_passage_time_stats_for_bins(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            trajectory_df=traj_df_tracked_sub,
            time_to_first_passage_col_name=time_to_first_passage_col_name,
            feature_column_names=list(DYNAMICS_COLUMN_NAMES),
        )

        # merge the grid and tracked first passage time stats dataframes
        fpt_stats_df = fpt_stats_df_grid.merge(
            fpt_stats_df_tracked,
            on=["bin_index"],
            suffixes=("_grid", "_tracked"),
            validate="one_to_one",
        )

        # check that the bin centers and edges are the same for the grid and tracked dataframes
        bin_centers_close = np.allclose(
            np.array(list(zip(*fpt_stats_df["bin_center_grid"], strict=True))),
            np.array(list(zip(*fpt_stats_df["bin_center_tracked"], strict=True))),
        )
        bin_edges_close = np.allclose(
            np.array(list(zip(*fpt_stats_df["bin_edges_grid"], strict=True))),
            np.array(list(zip(*fpt_stats_df["bin_edges_tracked"], strict=True))),
        )
        if not bin_centers_close or not bin_edges_close:
            error_message = (
                "Bin centers or edges are not the same for grid and tracked dataframes for "
                f"dataset {dataset_name} and fixed point {fp_idx}. This may indicate an issue "
                "with the binning or merging of the dataframes."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # drop the duplicate bin center and edge columns from one of the dataframes
        # since they are the same and rename the columns to remove the suffixes
        fpt_stats_df = fpt_stats_df.drop(columns=["bin_center_tracked", "bin_edges_tracked"])
        fpt_stats_df = fpt_stats_df.rename(
            columns={"bin_center_grid": "bin_center", "bin_edges_grid": "bin_edges"}
        )

        # 4. plot the cell FPT vs grid FPT data as a scatterplot with errors and a
        #    scatter with theta, r, rho as the axes and the FPT ratio as the color dimension
        # first the correlation scatter plots
        plot_first_passage_time_correlation(
            fixed_point_id=fp_idx,
            fixed_point_stability=fp_stability,
            dataset_config=dataset_config,
            first_passage_time_df=fpt_stats_df,
            stat_to_plot="mean",
            out_dir=out_dir,
        )
        plot_first_passage_time_correlation(
            fixed_point_id=fp_idx,
            fixed_point_stability=fp_stability,
            dataset_config=dataset_config,
            first_passage_time_df=fpt_stats_df,
            stat_to_plot="median",
            out_dir=out_dir,
        )
        # histograms don't really work for 4D data (theta, r, rho, and FPT ratio),
        # so we will use a 3D scatter with color-coded points instead
        plot_first_passage_time_3d_scatter(
            fixed_point_id=fp_idx,
            fixed_point_stability=fp_stability,
            dataset_config=dataset_config,
            first_passage_time_df=fpt_stats_df,
            fixed_points_df=fixed_points_df,
            stat_to_plot="mean",
            out_dir=out_dir,
        )
        plot_first_passage_time_3d_scatter(
            fixed_point_id=fp_idx,
            fixed_point_stability=fp_stability,
            dataset_config=dataset_config,
            first_passage_time_df=fpt_stats_df,
            fixed_points_df=fixed_points_df,
            stat_to_plot="median",
            out_dir=out_dir,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
