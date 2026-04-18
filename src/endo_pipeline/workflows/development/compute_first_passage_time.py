"""This workflow computes the time of first passage for each track in the dataset."""

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    # n_proc: int = 1,
):

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
        compute_first_passage_time_stats_for_bins,
        load_filtered_trajectory_df_for_first_passage_time_workflow,
    )
    from endo_pipeline.library.analyze.numerics.binning import adjust_limits_from_bin_size, get_bins
    from endo_pipeline.library.analyze.numerics.fixed_points import (
        load_fixed_points_dataframe_for_dataset,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_reshaped_vector_field_and_grid,
        load_drift_dataframe_for_dataset,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.settings import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.flow_field_3d import DATASET_COLLECTION_FOR_3D_DYNAMICS
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
        drift_df = load_drift_dataframe_for_dataset(dataset_name)
        drift_values, grid_points_1d = get_reshaped_vector_field_and_grid(
            flow_field_dataframe=drift_df, column_names=DYNAMICS_COLUMN_NAMES
        )
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

        # bin_centers_mesh = np.meshgrid(*bin_centers, indexing="ij")
        # bin_centers_all = list(zip(*[arr.ravel() for arr in bin_centers_mesh], strict=True))

        # 2. identify trajectories that pass a fixed point and filter df to only those trajectories
        # find if and when a trajectory reaches a fixed point
        for fp_idx in fixed_points_df.index:
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
                trajectory_df=traj_df_grid_sub,
                column=f"dist_from_fp_{fp_idx}",
                threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
            )
            traj_df_tracked_sub = add_first_passage_time_column(
                trajectory_df=traj_df_tracked_sub,
                column=f"dist_from_fp_{fp_idx}",
                threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
            )

            # trim all trajectories to only include timepoints prior to reaching the fixed point
            traj_df_grid_sub = traj_df_grid_sub[
                traj_df_grid_sub.apply(
                    lambda row: row[Column.TIMEPOINT] < row[f"first_passage_dist_from_fp_{fp_idx}"],
                    axis=1,
                )
            ]
            traj_df_tracked_sub = traj_df_tracked_sub[
                traj_df_tracked_sub.apply(
                    lambda row: row[Column.TIMEPOINT] < row[f"first_passage_dist_from_fp_{fp_idx}"],
                    axis=1,
                )
            ]

            # compute the time to the first passage time from each timepoint
            traj_df_grid_sub[f"time_to_fp_{fp_idx}"] = (
                traj_df_grid_sub[f"first_passage_dist_from_fp_{fp_idx}"]
                - traj_df_grid_sub[Column.TIMEPOINT]
            )
            traj_df_tracked_sub[f"time_to_fp_{fp_idx}"] = (
                traj_df_tracked_sub[f"first_passage_dist_from_fp_{fp_idx}"]
                - traj_df_tracked_sub[Column.TIMEPOINT]
            )

            # # find and remove the bin that corresponds to the fixed point location
            # # so that we don't analyze trajectories that start at the fixed point
            # fp_bin_index = []
            # for i, col in enumerate(DYNAMICS_COLUMN_NAMES):
            #     fp_bin_index.append(
            #         _get_index_from_value(fixed_points_df.iloc[fp_idx][col], bin_edges[i])
            #     )
            # fp_bin_center = [bin_centers[i][idx] for i, idx in enumerate(fp_bin_index)]

            # for b in bin_centers_all:
            #     if b == tuple(fp_bin_center):
            #         bin_centers_all.remove(b)

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
        #    heatmap with theta, r, rho as the axes and the FPT ratio as the color dimension

        from matplotlib import pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from scipy.stats import linregress

        stat = "mean"  # "50%"
        suffix = "_first_passage_time"
        metric = f"{stat}{suffix}"

        fpt_stats_df_no_nan = fpt_stats_df.dropna(subset=[f"{metric}_grid", f"{metric}_tracked"])
        # fpt_stats_df_no_nan = fpt_stats_df.dropna(
        #     subset=[f"std{suffix}_grid", f"std{suffix}_tracked"]
        # )
        linreg_results = linregress(
            x=fpt_stats_df_no_nan[f"{metric}_tracked"], y=fpt_stats_df_no_nan[f"{metric}_grid"]
        )

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.errorbar(
            x=fpt_stats_df[f"{metric}_grid"],
            y=fpt_stats_df[f"{metric}_tracked"],
            xerr=fpt_stats_df["std_first_passage_time_grid"],
            yerr=fpt_stats_df["std_first_passage_time_tracked"],
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            zorder=0,
        )
        ax.scatter(
            x=fpt_stats_df_no_nan[f"{metric}_grid"],
            y=fpt_stats_df_no_nan[f"{metric}_tracked"],
            color="black",
            edgecolor="white",
            lw=0.2,
        )
        ax.axline(xy1=(0, 0), slope=1, color="tab:red", linestyle="--", zorder=0)
        ax.axline(
            xy1=(0, linreg_results.intercept),
            slope=linreg_results.slope,
            color="tab:blue",
            linestyle="--",
            zorder=0,
        )
        ax_min = min((*ax.get_xlim(), *ax.get_ylim()))
        ax_max = max((*ax.get_xlim(), *ax.get_ylim()))
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3, 3.5), subplot_kw={"projection": "3d"})
        thetas, rs, rhos = zip(*fpt_stats_df_no_nan["bin_center"], strict=True)
        colors = np.log2(
            fpt_stats_df_no_nan[f"{metric}_tracked"] / fpt_stats_df_no_nan[f"{metric}_grid"]
        )
        scatter3d = ax.scatter(  # type: ignore[call-arg]
            xs=thetas,
            ys=rs,
            zs=rhos,
            c=colors,
            cmap="coolwarm_r",
            norm=TwoSlopeNorm(vcenter=0),
        )  # type: ignore[call-arg]
        ax.set_xlabel(get_label_for_column(Column.DiffAEData.POLAR_ANGLE))
        ax.set_ylabel(get_label_for_column(Column.DiffAEData.POLAR_RADIUS))
        ax.set_zlabel(get_label_for_column(Column.DiffAEData.PC3_FLIPPED))
        plt.tight_layout()
        cax = fig.add_axes([1.15, 0.2, 0.05, 0.6])  # type: ignore[call-overload]
        cbar = fig.colorbar(scatter3d, cax=cax)  # , ax=ax, shrink=0.5, aspect=5)
        # cbar.set_ticks([0.25, 0.5, 1, 2, 4])
        ax.scatter(*fixed_points_df.iloc[fp_idx][list(DYNAMICS_COLUMN_NAMES)].values, color="black", s=10, marker="*")  # type: ignore
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3, 3.5), subplot_kw={"projection": "3d"})
        thetas, rs, rhos = zip(*fpt_stats_df_no_nan["bin_center"], strict=True)
        colors = fpt_stats_df_no_nan[f"count{suffix}_grid"]
        scatter3d = ax.scatter(  # type: ignore[call-arg]
            xs=thetas,
            ys=rs,
            zs=rhos,
            c=colors,
            cmap="tab10",
            # norm=TwoSlopeNorm(vcenter=0),
        )  # type: ignore[call-arg]
        ax.set_xlabel(get_label_for_column(Column.DiffAEData.POLAR_ANGLE))
        ax.set_ylabel(get_label_for_column(Column.DiffAEData.POLAR_RADIUS))
        ax.set_zlabel(get_label_for_column(Column.DiffAEData.PC3_FLIPPED))
        plt.tight_layout()
        cax = fig.add_axes([1.15, 0.2, 0.05, 0.6])  # type: ignore[call-overload]
        cbar = fig.colorbar(scatter3d, cax=cax)  # , ax=ax, shrink=0.5, aspect=5)
        # cbar.set_ticks([0.25, 0.5, 1, 2, 4])
        ax.scatter(*fixed_points_df.iloc[fp_idx][list(DYNAMICS_COLUMN_NAMES)].values, color="black", s=10, marker="*")  # type: ignore
        plt.show()
        plt.close(fig)

        # sns.histplot(
        #     data=fpt_stats_df,
        #     x=f"{metric}_grid",
        #     y=f"{metric}_tracked",
        #     bins=fpt_stats_df["bin_edges"].tolist(),
        # )

        # for i in fixed_points_df.index:
        #     fixed_point_stability = fixed_points_df.loc[i, STABILITY_COLUMN_NAME]
        #     time_of_first_passage = []
        #     time_of_first_passage.append(
        #         get_first_passage_time(
        #             trajectory_df=traj_df_grid_sub,
        #             column=f"dist_from_fp_{i}_grid",
        #             threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
        #         )
        #     )
        #     time_of_first_passage.append(
        #         get_first_passage_time(
        #             trajectory_df=traj_df_tracked_sub,
        #             column=f"dist_from_fp_{i}_tracked",
        #             threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
        #         )
        #     )
        #     time_of_first_passage_df = pd.concat(time_of_first_passage, axis=1).reset_index()

        # plot_first_passage_time_histogram(
        #     fixed_point_id=i,
        #     fixed_point_stability=fixed_point_stability,
        #     dataset_config=dataset_config,
        #     time_of_first_passage_df=time_of_first_passage_df,
        #     out_dir=out_dir,
        #     crop_pattern=crop_pattern,
        # )

        # plot_first_passage_time_scatterplot(
        #     fixed_point_id=i,
        #     fixed_point_stability=fixed_point_stability,
        #     dataset_config=dataset_config,
        #     time_of_first_passage_df=time_of_first_passage_df,
        #     out_dir=out_dir,
        # )

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
