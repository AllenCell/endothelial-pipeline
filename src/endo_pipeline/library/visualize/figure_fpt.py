from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_binned_value
from endo_pipeline.library.analyze.numerics.binning import adjust_limits_from_bin_size, get_bins
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.track_integration import (
    add_distance_to_fixed_points_columns,
    add_first_passage_time_column,
    compute_first_passage_time_stats_for_bins,
    load_filtered_trajectory_df_for_first_passage_time_workflow,
    merge_grid_and_tracked_first_passage_time_stats_dfs,
)
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.settings import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    LONG_TRACK_THRESHOLD_LENGTH,
    POLAR_ANGLE_PERIOD,
    TIME_STEP_IN_HOURS,
)
from endo_pipeline.settings.figures import FONTSIZE_SMALL
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE


def generate_first_passage_time_example(
    dataset_name: str,
    example_fixed_point_index: int,
    example_tracked_crop_index: int | None,
    example_grid_crop_index: int | None,
    out_dir: Path | None = None,
    minimum_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
    fixed_point_radius_threshold: float = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
    min_num_traj_per_bin: int = 10,
) -> Path:

    if out_dir is None:
        out_dir = get_output_path(__file__)

    # load the dynamics features from the grid-based and track-based dataframes
    traj_df_grid = load_filtered_trajectory_df_for_first_passage_time_workflow(
        dataset_name,
        crop_pattern="grid",
        minimum_track_length=minimum_track_length,
    )
    traj_df_grid[Column.SegData.TIME_HRS] = traj_df_grid[Column.TIMEPOINT] * TIME_STEP_IN_HOURS

    traj_df_tracked = load_filtered_trajectory_df_for_first_passage_time_workflow(
        dataset_name,
        crop_pattern="tracked",
        minimum_track_length=minimum_track_length,
    )

    # load the flow field dictionaries and fixed points
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)
    traj_df_tracked[Column.SegData.TIME_HRS] = (
        traj_df_tracked[Column.TIMEPOINT] * TIME_STEP_IN_HOURS
    )

    # add the distances from the fixed points for the grid-based trajectories
    traj_df_grid = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_grid,
        fixed_point_df=fixed_points_df,
        trajectory_columns=list(DYNAMICS_COLUMN_NAMES),
        time_column=Column.SegData.TIME_HRS,
    )

    # add the distances from the fixed points for the track-based trajectories
    traj_df_tracked = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_tracked,
        fixed_point_df=fixed_points_df,
        trajectory_columns=list(DYNAMICS_COLUMN_NAMES),
        time_column=Column.SegData.TIME_HRS,
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

    bin_widths = tuple(float(bin_sizes[col]) for col in DYNAMICS_COLUMN_NAMES)
    bin_limits_list = [bin_limits[col] for col in DYNAMICS_COLUMN_NAMES]
    bin_edges, bin_centers = get_bins(bin_widths=bin_widths, bin_limits=bin_limits_list)

    # 2. mark each timepoint as "at the fixed point" if it is within the radius threshold,
    #    then propagate that flag to all timepoints in the trajectory using a groupby-transform
    #    so we can filter to only trajectories that ever reach the fixed point
    traj_df_grid[f"{Column.VectorField.IS_AT_FP_PREFIX}{example_fixed_point_index}"] = (
        traj_df_grid[f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{example_fixed_point_index}"]
        <= fixed_point_radius_threshold
    )
    traj_df_tracked[f"{Column.VectorField.IS_AT_FP_PREFIX}{example_fixed_point_index}"] = (
        traj_df_tracked[f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{example_fixed_point_index}"]
        <= fixed_point_radius_threshold
    )

    traj_df_grid[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{example_fixed_point_index}"] = (
        traj_df_grid.groupby(Column.CROP_INDEX)[
            f"{Column.VectorField.IS_AT_FP_PREFIX}{example_fixed_point_index}"
        ].transform(any)
    )
    traj_df_tracked[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{example_fixed_point_index}"] = (
        traj_df_tracked.groupby(Column.CROP_INDEX)[
            f"{Column.VectorField.IS_AT_FP_PREFIX}{example_fixed_point_index}"
        ].transform(any)
    )

    # keep only trajectories that eventually reach the fixed point
    traj_df_grid_sub = traj_df_grid[
        traj_df_grid[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{example_fixed_point_index}"]
    ]
    traj_df_tracked_sub = traj_df_tracked[
        traj_df_tracked[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{example_fixed_point_index}"]
    ]

    # compute the timepoint at which each trajectory first reaches a fixed point
    traj_df_grid_sub = add_first_passage_time_column(
        fixed_point_index=example_fixed_point_index,
        trajectory_df=traj_df_grid_sub,
        column=f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{example_fixed_point_index}",
        threshold=fixed_point_radius_threshold,
        time_column=Column.SegData.TIME_HRS,
    )
    traj_df_tracked_sub = add_first_passage_time_column(
        fixed_point_index=example_fixed_point_index,
        trajectory_df=traj_df_tracked_sub,
        column=f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{example_fixed_point_index}",
        threshold=fixed_point_radius_threshold,
        time_column=Column.SegData.TIME_HRS,
    )

    # 3. for each bin (across all steady-state timepoints), compute the mean,
    #    median, and standard deviation of first-passage times for the trajectories
    time_to_first_passage_col_name = (
        f"{Column.VectorField.TIME_TO_FP_PREFIX}{example_fixed_point_index}"
    )

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
    fpt_stats_df = merge_grid_and_tracked_first_passage_time_stats_dfs(
        fpt_stats_df_grid=fpt_stats_df_grid,
        fpt_stats_df_tracked=fpt_stats_df_tracked,
        dataset_name=dataset_name,
        fixed_point_index=example_fixed_point_index,
    )

    # plot the example tracks used to get the cell FPT vs grid FPT
    # remove bins that don't have enough trajectories in them from either the
    # grid or tracked trajectories
    fpt_stats_df = fpt_stats_df[
        fpt_stats_df["count_first_passage_time_grid"] >= min_num_traj_per_bin
    ]
    fpt_stats_df = fpt_stats_df[
        fpt_stats_df["count_first_passage_time_tracked"] >= min_num_traj_per_bin
    ]

    # pick the mean FPT metric and select the first well-populated bin as the
    # example bin to visualize
    # we take one representative trajectory from each crop pattern that starts inside that bin
    metric = "mean"
    suffix = Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX
    metric = f"{metric}{suffix}"

    # NaN values are unacceptable for the linear regression
    first_passage_time_df_no_nan = fpt_stats_df.copy().dropna(
        subset=[f"{metric}_grid", f"{metric}_tracked"]
    )
    # keep only the bins with the minimum number of tracks per bin in them
    first_passage_time_df_no_nan = first_passage_time_df_no_nan[
        first_passage_time_df_no_nan["count_first_passage_time_grid"] >= min_num_traj_per_bin
    ]
    first_passage_time_df_no_nan = first_passage_time_df_no_nan[
        first_passage_time_df_no_nan["count_first_passage_time_tracked"] >= min_num_traj_per_bin
    ]

    # get the center and edges of the first qualifying bin
    bin_center_1_bin, bin_edges_1_bin = first_passage_time_df_no_nan[
        [Column.VectorField.BIN_CENTER, Column.VectorField.BIN_EDGES]
    ].iloc[0]

    # filter trajectories to only those whose starting point falls in the example bin
    trajectory_df_tracked_one_bin = filter_dataframe_to_binned_value(
        dataframe=traj_df_tracked_sub,
        columns=list(DYNAMICS_COLUMN_NAMES),
        values=bin_center_1_bin,
        bin_edges=bin_edges,
    )
    trajectory_df_grid_one_bin = filter_dataframe_to_binned_value(
        dataframe=traj_df_grid_sub,
        columns=list(DYNAMICS_COLUMN_NAMES),
        values=bin_center_1_bin,
        bin_edges=bin_edges,
    )

    # take the first trajectory (by crop index) from each crop pattern in the
    # example bin; record its initial timepoint and feature-space position so
    # we can mark where it enters the bin
    for crop_id_tracked, df in trajectory_df_tracked_one_bin.groupby(Column.CROP_INDEX):
        df = df.sort_values(Column.TIMEPOINT)
        bin_tp_tracked, bin_theta_tracked, bin_r_tracked, bin_rho_tracked = (
            df.iloc[0][Column.TIMEPOINT],
            df.iloc[0][Column.DiffAEData.POLAR_ANGLE],
            df.iloc[0][Column.DiffAEData.POLAR_RADIUS],
            df.iloc[0][Column.DiffAEData.PC3_FLIPPED],
        )
        if example_tracked_crop_index is None or crop_id_tracked == example_tracked_crop_index:
            break

    for crop_id_grid, df in trajectory_df_grid_one_bin.groupby(Column.CROP_INDEX):
        df = df.sort_values(Column.TIMEPOINT)
        bin_tp_grid, bin_theta_grid, bin_r_grid, bin_rho_grid = (
            df.iloc[0][Column.TIMEPOINT],
            df.iloc[0][Column.DiffAEData.POLAR_ANGLE],
            df.iloc[0][Column.DiffAEData.POLAR_RADIUS],
            df.iloc[0][Column.DiffAEData.PC3_FLIPPED],
        )
        if example_grid_crop_index is None or crop_id_grid == example_grid_crop_index:
            break

    # extract the full trajectory from the bin entry point to the end of the track
    df_tracked_1_traj = traj_df_tracked_sub[
        (traj_df_tracked_sub[Column.CROP_INDEX] == crop_id_tracked)
        & (traj_df_tracked_sub[Column.TIMEPOINT] >= bin_tp_tracked)
    ]
    thetas_tracked, rs_tracked, rhos_tracked = zip(
        *df_tracked_1_traj[
            [
                Column.DiffAEData.POLAR_ANGLE,
                Column.DiffAEData.POLAR_RADIUS,
                Column.DiffAEData.PC3_FLIPPED,
            ]
        ].values,
        strict=True,
    )
    df_grid_1_traj = traj_df_grid_sub[
        (traj_df_grid_sub[Column.CROP_INDEX] == crop_id_grid)
        & (traj_df_grid_sub[Column.TIMEPOINT] >= bin_tp_grid)
    ]
    thetas_grid, rs_grid, rhos_grid = zip(
        *df_grid_1_traj[
            [
                Column.DiffAEData.POLAR_ANGLE,
                Column.DiffAEData.POLAR_RADIUS,
                Column.DiffAEData.PC3_FLIPPED,
            ]
        ].values,
        strict=True,
    )
    # unwrap the polar angle for the grid trajectory to avoid discontinuous jumps
    # across the 0/2*pi boundary when plotting
    polar_angle_period = POLAR_ANGLE_PERIOD
    thetas_grid_unwrapped = np.unwrap(thetas_grid, period=polar_angle_period)
    thetas_tracked_unwrapped = np.unwrap(thetas_tracked, period=polar_angle_period)

    # build the 8 corner vertices of the example bin (a rectangular cuboid in
    # feature space), then identify the 12 axis-aligned edges by keeping only
    # vertex pairs whose distance equals one of the three bin side lengths
    xs, ys, zs = np.meshgrid(*bin_edges_1_bin)
    xs = xs.ravel()
    ys = ys.ravel()
    zs = zs.ravel()

    vertices = list(zip(xs, ys, zs, strict=True))
    edges = []
    for v1, v2 in combinations(vertices, r=2):
        edge_length = np.linalg.norm(np.array(v1) - np.array(v2))
        if np.isclose(edge_length, bin_sizes[Column.DiffAEData.POLAR_ANGLE]):
            edges.append((v1, v2))
        elif np.isclose(edge_length, bin_sizes[Column.DiffAEData.POLAR_RADIUS]):
            edges.append((v1, v2))
        elif np.isclose(edge_length, bin_sizes[Column.DiffAEData.PC3_FLIPPED]):
            edges.append((v1, v2))

    # plot the tracked and grid trajectories in the 3D feature space with the
    # fixed point, bin edges, and bin start points
    track_alpha = 0.4
    fig = plt.figure(figsize=(2, 2))
    ax: Axes3D = fig.add_subplot(projection="3d")
    ax.plot(
        xs=thetas_tracked_unwrapped,
        ys=rs_tracked,
        zs=rhos_tracked,
        ls="-",
        lw=1,
        marker=".",
        c="tab:red",
        alpha=track_alpha,
        label="tracked",
    )
    ax.plot(
        xs=thetas_grid_unwrapped,
        ys=rs_grid,
        zs=rhos_grid,
        ls="-",
        lw=1,
        marker="d",
        c="tab:blue",
        alpha=track_alpha,
        label="grid",
    )
    # plot the FPT start points as black markers with no fill
    ax.scatter(
        bin_theta_tracked,
        bin_r_tracked,
        bin_rho_tracked,
        edgecolors="black",
        facecolors="none",
        lw=1,
        s=5,
        marker="o",
    )
    ax.scatter(
        bin_theta_grid,
        bin_r_grid,
        bin_rho_grid,
        edgecolors="black",
        facecolors="none",
        lw=1,
        s=5,
        marker="d",
    )
    # draw cube around bin edges
    for e_xyz in edges:
        ax.plot(*list(zip(*e_xyz, strict=True)), ls="-", lw=0.5, c="black", alpha=0.7)
    # plot the fixed point in the 3D space as a black star
    fp_dynamic_cols = [str(col) for col in DYNAMICS_COLUMN_NAMES]
    ax.scatter(
        *fixed_points_df.loc[example_fixed_point_index][fp_dynamic_cols].values,
        color="black",
        s=15,
        marker="*",
        label="fixed point",
    )
    # plot a sphere around the fixed point with radius equal to the fixed_point_radius_threshold
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = fixed_points_df.loc[example_fixed_point_index][
        Column.DiffAEData.POLAR_ANGLE
    ] + fixed_point_radius_threshold * np.outer(np.cos(u), np.sin(v))
    y = fixed_points_df.loc[example_fixed_point_index][
        Column.DiffAEData.POLAR_RADIUS
    ] + fixed_point_radius_threshold * np.outer(np.sin(u), np.sin(v))
    z = fixed_points_df.loc[example_fixed_point_index][
        Column.DiffAEData.PC3_FLIPPED
    ] + fixed_point_radius_threshold * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="black", alpha=0.1)
    ax.set_aspect("equal")
    # make the minor ticks on each axis correspond to the bin edges
    theta_lims = ax.get_xlim()
    r_lims = ax.get_ylim()
    rho_lims = ax.get_zlim()
    # extend theta bins to account for angle wrapping
    bin_edges_theta_plot = np.concatenate((-1 * bin_edges[0][1:], bin_edges[0]))
    bin_edges_r_plot = bin_edges[1]
    bin_edges_rho_plot = bin_edges[2]
    theta_minor_ticks = bin_edges_theta_plot[
        (bin_edges_theta_plot > theta_lims[0]) & (bin_edges_theta_plot < theta_lims[1])
    ]

    r_minor_ticks = bin_edges_r_plot[
        (bin_edges_r_plot > r_lims[0]) & (bin_edges_r_plot < r_lims[1])
    ]
    # r minor ticks are too close together, so skip every other LABEL (NOT TICK)
    r_minor_ticklabels = [
        f"{tick:.2f}" if tick in r_minor_ticks[::2] else "" for tick in r_minor_ticks
    ]
    rho_minor_ticks = bin_edges_rho_plot[
        (bin_edges_rho_plot > rho_lims[0]) & (bin_edges_rho_plot < rho_lims[1])
    ]
    ax.set_xticks(theta_minor_ticks)
    ax.set_yticks(r_minor_ticks)
    ax.set_zticks(rho_minor_ticks)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax.set_yticklabels(r_minor_ticklabels)
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    # make the axes labels pretty
    ax.tick_params(axis="x", labelsize=FONTSIZE_SMALL, rotation=45, pad=4)
    plt.setp(ax.get_xticklabels(), va="bottom", ha="center")
    ax.tick_params(axis="y", labelsize=FONTSIZE_SMALL, rotation=-15, pad=-2)
    plt.setp(ax.get_yticklabels(), va="center", ha="left")
    ax.tick_params(axis="z", labelsize=FONTSIZE_SMALL, pad=-4)
    plt.setp(ax.get_zticklabels(), va="top", ha="left")
    ax.set_xlabel(get_label_for_column(Column.DiffAEData.POLAR_ANGLE), loc="center", labelpad=-4)
    ax.set_ylabel(
        get_label_for_column(Column.DiffAEData.POLAR_RADIUS), loc="center", labelpad=0
    )  # 5)
    ax.set_zlabel(get_label_for_column(Column.DiffAEData.PC3_FLIPPED), labelpad=-1)  # 5)

    # adjust the focal length of the 3D plot so that depth is easier to perceive
    ax.set_proj_type("persp", focal_length=0.3)
    ax.view_init(elev=30, azim=-40)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    ax_width = 0.7
    ax_height = 0.7
    ax.set_position([(1 - ax_width) / 2, (1 - ax_height) / 2, ax_width, ax_height])

    filename = f"{dataset_name}_FPT_fp_{example_fixed_point_index}_mean_3d_scatter"
    save_plot_to_path(fig, out_dir, filename, file_format=".svg")

    return out_dir / f"{filename}.svg"
