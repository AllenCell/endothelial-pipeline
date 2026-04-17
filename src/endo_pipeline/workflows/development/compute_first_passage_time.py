"""This workflow computes the time of first passage for each track in the dataset."""

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
):

    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs.dataset_config_io import (
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        get_drift_df,
        get_drift_values_and_grid_from_drift_df,
        get_fixed_points_df,
    )
    from endo_pipeline.library.analyze.integration.track_integration import (
        add_distance_to_fixed_points_columns,
        get_first_passage_time,
        load_filtered_trajectory_df_for_first_passage_time_workflow,
    )
    from endo_pipeline.library.analyze.numerics.binning import adjust_limits_from_bin_size, get_bins
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_first_passage_time_scatterplot,
    )
    from endo_pipeline.settings import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.flow_field_3d import DATASET_COLLECTION_FOR_3D_DYNAMICS
    from endo_pipeline.settings.flow_field_dataframes import STABILITY_COLUMN_NAME
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE

    logger = logging.getLogger(__name__)

    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)

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
            dataset_name, crop_pattern="grid"
        )
        traj_df_tracked = load_filtered_trajectory_df_for_first_passage_time_workflow(
            dataset_name, crop_pattern="tracked"
        )

        # load the flow field dictionaries and fixed points
        # NOTE - might keep the flow field dictionary to create example plots
        drift_df = get_drift_df(dataset_name)
        drift_values, grid_points_1d = get_drift_values_and_grid_from_drift_df(
            flow_field_dataframe=drift_df, column_names=DYNAMICS_COLUMN_NAMES
        )
        fixed_points_df = get_fixed_points_df(dataset_name)

        if fixed_points_df.empty:
            logger.warning(f"No fixed points found for dataset {dataset_name}, skipping dataset.")
            continue

        # 1. identify trajectories that pass a fixed point and filter df to only those trajectories
        # add the distances from the fixed points for the grid-based trajectories
        traj_df_grid = add_distance_to_fixed_points_columns(
            trajectory_df=traj_df_grid,
            fixed_point_df=fixed_points_df,
            trajectory_columns=DYNAMICS_COLUMN_NAMES,
            column_suffix="grid",
        )

        # add the distances from the fixed points for the track-based trajectories
        traj_df_tracked = add_distance_to_fixed_points_columns(
            trajectory_df=traj_df_tracked,
            fixed_point_df=fixed_points_df,
            trajectory_columns=DYNAMICS_COLUMN_NAMES,
            column_suffix="tracked",
        )

        # find if and when a trajectory reaches a fixed point
        for fpt_idx in fixed_points_df.index:
            traj_df_grid[f"has_reached_fp_{fpt_idx}_grid"] = (
                traj_df_grid[f"dist_from_fp_{fpt_idx}_grid"]
                <= MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
            )
            traj_df_tracked[f"has_reached_fp_{fpt_idx}_tracked"] = (
                traj_df_tracked[f"dist_from_fp_{fpt_idx}_tracked"]
                <= MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
            )

            traj_df_grid[f"traj_reached_fp_{fpt_idx}_grid"] = traj_df_grid.groupby(
                Column.CROP_INDEX
            )[f"has_reached_fp_{fpt_idx}_grid"].transform(any)
            traj_df_tracked[f"traj_reached_fp_{fpt_idx}_tracked"] = traj_df_tracked.groupby(
                Column.CROP_INDEX
            )[f"has_reached_fp_{fpt_idx}_tracked"].transform(any)

            traj_df_grid_sub = traj_df_grid[traj_df_grid[f"traj_reached_fp_{fpt_idx}_grid"]]
            traj_df_tracked_sub = traj_df_tracked[
                traj_df_tracked[f"traj_reached_fp_{fpt_idx}_tracked"]
            ]

        # 2. bin (theta, r, rho) feature space
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

        bin_centers_mesh = np.meshgrid(*bin_centers, indexing="ij")
        bin_centers_all = list(zip(*[arr.ravel() for arr in bin_centers_mesh], strict=True))

        # 3. for each bin (across all steady-state timepoints), compute the mean,
        #    median, and standard deviation of first-passage times for the trajectories
        # NOTE THIS PART SHOULD BE MULTIPROCESSED: 1 CORE FOR EACH BIN

        # 4. plot the cell FPT vs grid FPT data as a scatterplot with errors and a
        #    heatmap with theta, r, rho as the axes and the FPT ratio as the color dimension

        for i in fixed_points_df.index:
            fixed_point_stability = fixed_points_df.loc[i, STABILITY_COLUMN_NAME]
            time_of_first_passage = []
            time_of_first_passage.append(
                get_first_passage_time(
                    trajectory_df=traj_df_grid_sub,
                    column=f"dist_from_fp_{i}_grid",
                    threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
                )
            )
            time_of_first_passage.append(
                get_first_passage_time(
                    trajectory_df=traj_df_tracked_sub,
                    column=f"dist_from_fp_{i}_tracked",
                    threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
                )
            )
            time_of_first_passage_df = pd.concat(time_of_first_passage, axis=1).reset_index()

            # plot_first_passage_time_histogram(
            #     fixed_point_id=i,
            #     fixed_point_stability=fixed_point_stability,
            #     dataset_config=dataset_config,
            #     time_of_first_passage_df=time_of_first_passage_df,
            #     out_dir=out_dir,
            #     crop_pattern=crop_pattern,
            # )

        plot_first_passage_time_scatterplot(
            fixed_point_id=i,
            fixed_point_stability=fixed_point_stability,
            dataset_config=dataset_config,
            time_of_first_passage_df=time_of_first_passage_df,
            out_dir=out_dir,
        )

        # bin_sizes = [
        #     {"num_bins_polar_theta": 12, "num_bins_polar_r": 12, "num_bins_rho": 12},
        #     {"num_bins_polar_theta": 8, "num_bins_polar_r": 8, "num_bins_rho": 8},
        #     {"num_bins_polar_theta": 6, "num_bins_polar_r": 6, "num_bins_rho": 6},
        #     {"num_bins_polar_theta": 12, "num_bins_polar_r": 12, "num_bins_rho": 1},
        # ]
        # for bins in bin_sizes:
        #     plot_initial_conditions_histogram(
        #         df_first_timepoint=trajectories_df_t_init,
        #         num_bins_polar_theta=bins["num_bins_polar_theta"],
        #         num_bins_polar_r=bins["num_bins_polar_r"],
        #         num_bins_rho=bins["num_bins_rho"],
        #         out_dir=out_dir,
        #     )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
