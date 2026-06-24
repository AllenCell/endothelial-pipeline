from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from endo_pipeline.io import save_plot_to_path
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
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    LONG_TRACK_THRESHOLD_LENGTH,
    POLAR_ANGLE_PERIOD,
    TIME_STEP_IN_HOURS,
)
from endo_pipeline.settings.figures import FONTSIZE_SMALL
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode


class _StoppingRadiusSphereHandle:
    """Dummy handle used to dispatch to _HandlerSphere in ax.legend()."""


class _HandlerSphere(HandlerBase):
    """Renders the stopping-radius legend entry as a shaded 3-D sphere.

    Draws a filled grey disc with a small white specular highlight offset
    toward the upper-left, giving the classic illustration of a sphere.
    """

    def create_artists(
        self, _legend, _orig_handle, xdescent, ydescent, width, height, _fontsize, trans
    ):
        """
        Create a legend handle artist for the stopping radius sphere.

        Parameters
        ----------
        _legend
            The Legend object to which the handler is being applied (not used),
            kept for API compatibility with HandlerBase.
        _orig_handle
            The original handle (the object being represented in the legend, not
            used), kept for API compatibility with HandlerBase.
        xdescent
            The horizontal space to reserve for the handle.
        _ydescent
            The vertical space to reserve for the handle (not used), kept for
            API compatibility with HandlerBase.
        width
            The total width of the area allocated for the handle.
        height
            The total height of the area allocated for the handle.
        _fontsize
            The font size of the legend text (not used), kept for API
            compatibility with HandlerBase.
        trans
            The transformation to apply to the created artists to position them
            correctly in the legend.

        Returns
        -------
        :
            List of artists (sphere body and highlight) to be added to the legend.
        """
        cx = (width - xdescent) / 2
        cy = (height - ydescent) / 2
        r = min(width, height) / 2 * 2.0

        body = Circle(
            (cx, cy),
            r,
            facecolor="grey",
            alpha=0.5,
            edgecolor="black",
            linewidth=0.25,
            transform=trans,
        )
        highlight = Circle(
            (cx - r * 0.28, cy + r * 0.28),
            r * 0.3,
            facecolor="white",
            alpha=0.55,
            edgecolor="none",
            transform=trans,
        )
        return [body, highlight]


def _load_trajectory_dataframes(
    dataset_name: str, minimum_track_length: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process feature dataframes for first passage time analysis and
    visualization.

    First, calls the method
    :py:func:`~endo_pipeline.library.analyze.track_integration.load_filtered_trajectory_df_for_first_passage_time_workflow`
    to load the trajectory dataframes for both grid-based and cell-centered
    trajectories, filtered to only include trajectories of at least the
    specified minimum track length.

    Then adds columns for the distance of each trajectory timepoint to each
    fixed point, which will be used for filtering to trajectories that reach the
    fixed point and for computing first passage times.

    Parameters
    ----------
    dataset_name
        Name of the dataset to load trajectory data for.
    minimum_track_length
        Minimum track length to filter trajectories by when loading the
        trajectory dataframes.

    Returns
    -------
    :
        Tuple of (grid-based trajectory dataframe, cell-centered trajectory
        dataframe, fixed points dataframe) with distance to fixed point columns
        added to the trajectory dataframes.
    """
    traj_df_grid = load_filtered_trajectory_df_for_first_passage_time_workflow(
        dataset_name,
        patch_type="grid_based",
        minimum_track_length=minimum_track_length,
    )
    traj_df_grid[Column.SegData.TIME_HRS] = traj_df_grid[Column.TIMEPOINT] * TIME_STEP_IN_HOURS

    traj_df_tracked = load_filtered_trajectory_df_for_first_passage_time_workflow(
        dataset_name,
        patch_type="cell_centered",
        minimum_track_length=minimum_track_length,
    )
    traj_df_tracked[Column.SegData.TIME_HRS] = (
        traj_df_tracked[Column.TIMEPOINT] * TIME_STEP_IN_HOURS
    )

    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)

    traj_df_grid = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_grid,
        fixed_point_df=fixed_points_df,
        trajectory_columns=list(DYNAMICS_COLUMN_NAMES),
        time_column=Column.SegData.TIME_HRS,
    )
    traj_df_tracked = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_tracked,
        fixed_point_df=fixed_points_df,
        trajectory_columns=list(DYNAMICS_COLUMN_NAMES),
        time_column=Column.SegData.TIME_HRS,
    )

    return traj_df_grid, traj_df_tracked, fixed_points_df


def _filter_to_first_passage_trajectories(
    traj_df_grid: pd.DataFrame,
    traj_df_tracked: pd.DataFrame,
    fixed_point_index: int,
    fixed_point_radius_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the trajectory dataframes to only include trajectories that reach
    the stable fixed point.

    Parameters
    ----------
    traj_df_grid
        Dataframe of grid-based trajectories with distance to fixed point columns.
    traj_df_tracked
        Dataframe of cell-centered trajectories with distance to fixed point columns.
    fixed_point_index
        Index of the fixed point to filter trajectories by.
    fixed_point_radius_threshold
        Distance threshold from the fixed point below which a trajectory timepoint
        is considered to have reached the fixed point.

    Returns
    -------
    :
        Tuple of (filtered grid-based trajectory dataframe, filtered cell-centered
        trajectory dataframe) that only include trajectories that reach the
        fixed point.
    """
    # Mark each timepoint as "at the fixed point" if it is within the radius
    # threshold, then propagate that flag to all timepoints in the trajectory
    # using a groupby-transform so we can filter to only trajectories that ever
    # reach the fixed point
    traj_df_grid[f"{Column.VectorField.IS_AT_FP_PREFIX}{fixed_point_index}"] = (
        traj_df_grid[f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fixed_point_index}"]
        <= fixed_point_radius_threshold
    )
    traj_df_tracked[f"{Column.VectorField.IS_AT_FP_PREFIX}{fixed_point_index}"] = (
        traj_df_tracked[f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fixed_point_index}"]
        <= fixed_point_radius_threshold
    )

    traj_df_grid[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fixed_point_index}"] = (
        traj_df_grid.groupby(Column.CROP_INDEX)[
            f"{Column.VectorField.IS_AT_FP_PREFIX}{fixed_point_index}"
        ].transform(any)
    )
    traj_df_tracked[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fixed_point_index}"] = (
        traj_df_tracked.groupby(Column.CROP_INDEX)[
            f"{Column.VectorField.IS_AT_FP_PREFIX}{fixed_point_index}"
        ].transform(any)
    )

    # Keep only trajectories that eventually reach the fixed point
    traj_df_grid_sub = traj_df_grid[
        traj_df_grid[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fixed_point_index}"]
    ]
    traj_df_tracked_sub = traj_df_tracked[
        traj_df_tracked[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fixed_point_index}"]
    ]

    return traj_df_grid_sub, traj_df_tracked_sub


def _compute_filtered_fpt_stats(
    traj_df_grid_sub: pd.DataFrame,
    traj_df_tracked_sub: pd.DataFrame,
    dataset_name: str,
    fixed_point_index: int,
    bin_edges: list[np.ndarray],
    bin_centers: list[np.ndarray],
    min_num_traj_per_bin: int,
) -> pd.DataFrame:
    """
    Compute first passage time statisics per bin.

    **Output dataframe**

    This method outputs a dataframe with mean, median, and standard deviation of
    first-passage times for the trajectories that start in each bin, for both
    grid-based and cell-centered trajectories. It also includes the count of
    trajectories that start in each bin for both grid-based and cell-centered
    trajectories.

    Parameters
    ----------
    traj_df_grid_sub
        Dataframe of grid-based trajectories that reach the fixed point, with
        distance to fixed point columns and first passage time column.
    traj_df_tracked_sub
        Dataframe of cell-centered trajectories that reach the fixed point, with
        distance to fixed point columns and first passage time column.
    dataset_name
        Name of the dataset being analyzed, used for labeling the output
        dataframe.
    fixed_point_index
        Index of the fixed point that the trajectories reach, used for labeling
        the first passage time column in the output dataframe.
    bin_edges
        Edges of the bins in feature space to compute statistics for.
    bin_centers
        Centers of the bins in feature space to compute statistics for.
    min_num_traj_per_bin
        Minimum number of trajectories that must start in a bin for that bin to
        be included in the output dataframe.

    Returns
    -------
    :
        Dataframe with first passage time statistics.
    """
    # For each bin (across all steady-state timepoints), compute the mean,
    # median, and standard deviation of first-passage times for the trajectories
    time_to_first_passage_col_name = f"{Column.VectorField.TIME_TO_FP_PREFIX}{fixed_point_index}"

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
        fixed_point_index=fixed_point_index,
    )

    # remove bins that don't have enough trajectories in them from either the
    # grid or tracked trajectories
    fpt_stats_df = fpt_stats_df[
        fpt_stats_df["count_first_passage_time_grid"] >= min_num_traj_per_bin
    ]
    fpt_stats_df = fpt_stats_df[
        fpt_stats_df["count_first_passage_time_tracked"] >= min_num_traj_per_bin
    ]

    return fpt_stats_df


def _select_example_bin(
    fpt_stats_df: pd.DataFrame, min_num_traj_per_bin: int
) -> tuple[np.ndarray, list]:
    """
    Select an example bin to visualize trajectories from based on the first
    passage time statistics dataframe.
    """
    # pick the mean FPT metric and select the first well-populated bin as the
    # example bin to visualize
    # we take one representative trajectory from each crop pattern that starts inside that bin
    metric = f"mean{Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX}"

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
    example_bin_center, example_bin_edges = first_passage_time_df_no_nan[
        [Column.VectorField.BIN_CENTER, Column.VectorField.BIN_EDGES]
    ].iloc[0]
    return example_bin_center, example_bin_edges


def _select_example_bin_trajectories(
    example_bin_center: np.ndarray,
    bin_edges: list[np.ndarray],
    traj_df_grid_sub: pd.DataFrame,
    traj_df_tracked_sub: pd.DataFrame,
    example_tracked_crop_index: int | None,
    example_grid_crop_index: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter dataframes to only include a single trajectory from each crop pattern
    that starts in the example bin, and filter to just the timepoints from that
    trajectory that start in the bin and continue until the end of the track.
    """
    # filter trajectories to only those whose starting point falls in the example bin
    trajectory_df_tracked_one_bin = filter_dataframe_to_binned_value(
        dataframe=traj_df_tracked_sub,
        columns=list(DYNAMICS_COLUMN_NAMES),
        values=example_bin_center,
        bin_edges=bin_edges,
    )
    trajectory_df_grid_one_bin = filter_dataframe_to_binned_value(
        dataframe=traj_df_grid_sub,
        columns=list(DYNAMICS_COLUMN_NAMES),
        values=example_bin_center,
        bin_edges=bin_edges,
    )

    # take the first trajectory (by crop index) from each patch type in the
    # example bin; record its initial timepoint and feature-space position so
    # we can mark where it enters the bin
    for crop_id_tracked, df in trajectory_df_tracked_one_bin.groupby(Column.CROP_INDEX):
        df = df.sort_values(Column.TIMEPOINT)
        bin_tp_tracked = df.iloc[0][Column.TIMEPOINT]
        if example_tracked_crop_index is None or crop_id_tracked == example_tracked_crop_index:
            break

    for crop_id_grid, df in trajectory_df_grid_one_bin.groupby(Column.CROP_INDEX):
        df = df.sort_values(Column.TIMEPOINT)
        bin_tp_grid = df.iloc[0][Column.TIMEPOINT]
        if example_grid_crop_index is None or crop_id_grid == example_grid_crop_index:
            break

    # extract the full trajectory from the bin entry point to the end of the track
    example_traj_df_tracked = traj_df_tracked_sub[
        (traj_df_tracked_sub[Column.CROP_INDEX] == crop_id_tracked)
        & (traj_df_tracked_sub[Column.TIMEPOINT] >= bin_tp_tracked)
    ]
    example_traj_df_grid = traj_df_grid_sub[
        (traj_df_grid_sub[Column.CROP_INDEX] == crop_id_grid)
        & (traj_df_grid_sub[Column.TIMEPOINT] >= bin_tp_grid)
    ]

    return example_traj_df_grid, example_traj_df_tracked


def _build_bin_cuboid_edges(example_bin_edges: list, bin_sizes: dict):
    # build the 8 corner vertices of the example bin (a rectangular cuboid in
    # feature space), then identify the 12 axis-aligned edges by keeping only
    # vertex pairs whose distance equals one of the three bin side lengths
    xs, ys, zs = np.meshgrid(*example_bin_edges)
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

    return edges


def _create_fpt_schematic_figure(
    figure_size: tuple[float, float],
    example_traj_df_tracked: pd.DataFrame,
    example_traj_df_grid: pd.DataFrame,
    fixed_points_df: pd.DataFrame,
    example_fixed_point_index: int,
    fixed_point_radius_threshold: float,
    example_bin_cuboid_edges: list[tuple[np.ndarray, np.ndarray]],
    grid_bin_edges: list[np.ndarray],
) -> plt.Figure:
    """
    Create a schematic figure illustrating the trajectories used for first
    passage time visualization, with the fixed point and bin edges overlaid in
    the 3D feature space.
    """

    # unpack dataframe to numpy arrays
    thetas_tracked, rs_tracked, rhos_tracked = zip(
        *example_traj_df_tracked[
            [
                Column.DiffAEData.POLAR_ANGLE,
                Column.DiffAEData.POLAR_RADIUS,
                Column.DiffAEData.PC3_FLIPPED,
            ]
        ].values,
        strict=True,
    )
    thetas_grid, rs_grid, rhos_grid = zip(
        *example_traj_df_grid[
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
    thetas_grid_unwrapped = np.unwrap(thetas_grid, period=POLAR_ANGLE_PERIOD)
    thetas_tracked_unwrapped = np.unwrap(thetas_tracked, period=POLAR_ANGLE_PERIOD)

    # plot the tracked and grid trajectories in the 3D feature space with the
    # fixed point, bin edges, and bin start points
    fig = plt.figure(figsize=figure_size)
    ax: Axes3D = fig.add_subplot(projection="3d")
    grid_color = "tab:blue"
    grid_marker = "d"
    cell_centric_color = "tab:red"
    cell_centric_marker_line = "."
    cell_centric_marker_scatter = "o"
    track_alpha = 0.7

    # plot the full trajectories as lines with markers at each timepoint,
    # colored by crop pattern
    ax.plot(
        xs=thetas_grid_unwrapped,
        ys=rs_grid,
        zs=rhos_grid,
        ls="-",
        lw=1,
        marker=grid_marker,
        markersize=1,
        c=grid_color,
        alpha=track_alpha,
        label="grid-based trajectory",
    )
    ax.plot(
        xs=thetas_tracked_unwrapped,
        ys=rs_tracked,
        zs=rhos_tracked,
        ls="-",
        lw=1,
        marker=cell_centric_marker_line,
        markersize=3,
        c=cell_centric_color,
        alpha=track_alpha,
        label="cell-centered trajectory",
    )

    # plot the FPT start points as larger markers with black outline
    ax.scatter(
        thetas_grid_unwrapped[0],
        rs_grid[0],
        rhos_grid[0],
        facecolors=grid_color,
        alpha=track_alpha,
        s=15,
        marker=grid_marker,
    )
    ax.scatter(
        thetas_grid_unwrapped[0],
        rs_grid[0],
        rhos_grid[0],
        edgecolors="black",
        facecolors="none",
        alpha=1.0,
        lw=0.75,
        s=15,
        marker=grid_marker,
    )
    ax.scatter(
        thetas_tracked_unwrapped[0],
        rs_tracked[0],
        rhos_tracked[0],
        facecolors=cell_centric_color,
        alpha=track_alpha,
        s=15,
        marker=cell_centric_marker_scatter,
    )
    ax.scatter(
        thetas_tracked_unwrapped[0],
        rs_tracked[0],
        rhos_tracked[0],
        edgecolors="black",
        facecolors="none",
        alpha=1.0,
        lw=0.75,
        s=15,
        marker=cell_centric_marker_scatter,
    )

    # draw cube around bin edges
    for e_xyz in example_bin_cuboid_edges:
        ax.plot(*list(zip(*e_xyz, strict=True)), ls="-", lw=0.5, c="black", alpha=0.9, zorder=1)

    # plot the fixed point in the 3D space using consistent marker formatting
    fp_dynamic_cols = [str(col) for col in DYNAMICS_COLUMN_NAMES]
    col_labels = [(COLUMN_METADATA[col].label or col) for col in fp_dynamic_cols]
    fixed_point_label = f"({col_labels[0]}$^*$, {col_labels[1]}$^*$, {col_labels[2]}$^*$)"
    ax.scatter(
        *fixed_points_df.loc[example_fixed_point_index][fp_dynamic_cols].values,
        marker=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
        color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
        s=15,
        label=fixed_point_label,
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
    ax.plot_surface(x, y, z, color="black", alpha=0.1, edgecolor="k", linewidth=0.25)
    ax.set_aspect("equal")

    # make the minor ticks on each axis correspond to the bin edges
    theta_lims = ax.get_xlim()
    r_lims = ax.get_ylim()
    rho_lims = ax.get_zlim()

    # extend theta bins to account for angle wrapping
    bin_edges_theta_plot = np.concatenate((-1 * grid_bin_edges[0][1:], grid_bin_edges[0]))
    bin_edges_r_plot = grid_bin_edges[1]
    bin_edges_rho_plot = grid_bin_edges[2]
    theta_minor_ticks = bin_edges_theta_plot[
        (bin_edges_theta_plot > theta_lims[0]) & (bin_edges_theta_plot < theta_lims[1])
    ]
    theta_minor_ticklabels = [0, "", f"{Unicode.PI}/6"]
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
    ax.set_xticklabels(theta_minor_ticklabels)
    ax.set_yticklabels(r_minor_ticklabels)
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    # make the axes labels pretty
    ax.tick_params(axis="x", labelsize=FONTSIZE_SMALL, rotation=0, pad=-2)
    plt.setp(ax.get_xticklabels(), va="bottom", ha="center")
    ax.tick_params(axis="y", labelsize=FONTSIZE_SMALL, rotation=-15, pad=-2)
    plt.setp(ax.get_yticklabels(), va="center", ha="left")
    ax.tick_params(axis="z", labelsize=FONTSIZE_SMALL, pad=-4)
    plt.setp(ax.get_zticklabels(), va="top", ha="left")
    ax.set_xlabel(get_label_for_column(Column.DiffAEData.POLAR_ANGLE), loc="center", labelpad=-8)
    ax.set_ylabel(get_label_for_column(Column.DiffAEData.POLAR_RADIUS), loc="center", labelpad=0)
    ax.set_zlabel(get_label_for_column(Column.DiffAEData.PC3_FLIPPED), labelpad=-1)

    # adjust the focal length of the 3D plot so that depth is easier to perceive
    ax.set_proj_type("persp", focal_length=0.3)
    ax.view_init(elev=30, azim=-35)
    ax_width = 0.7
    ax_height = 0.7
    ax.set_position([(1 - ax_width) / 2, (1 - ax_height) / 2, ax_width, ax_height])

    # add legend
    handles, labels = ax.get_legend_handles_labels()
    starting_point_handle_grid = Line2D(
        [0],
        [0],
        linestyle="none",
        marker=grid_marker,
        markeredgecolor="black",
        markerfacecolor="none",
        markeredgewidth=0.75,
        markersize=4,
    )
    handles.insert(-1, starting_point_handle_grid)
    labels.insert(-1, "grid-based trajectory\n starting point from bin")
    starting_point_handle_cell = Line2D(
        [0],
        [0],
        linestyle="none",
        marker=cell_centric_marker_scatter,
        markeredgecolor="black",
        markerfacecolor="none",
        markeredgewidth=0.75,
        markersize=4,
    )
    handles.insert(-1, starting_point_handle_cell)
    labels.insert(-1, "cell-centered trajectory\n starting point from bin")
    sphere_handle = _StoppingRadiusSphereHandle()
    handles.append(sphere_handle)
    labels.append("fixed point\nstopping radius")
    ax.legend(
        handles,
        labels,
        ncols=1,
        loc="upper left",
        bbox_to_anchor=(1.25, 0.85),
        handler_map={_StoppingRadiusSphereHandle: _HandlerSphere()},
    )

    return fig


@figure_panel("Plot example trajectories used to compute MFPTs (cell-centered vs. grid-based).")
def generate_first_passage_time_example(
    dataset_name: str,
    example_fixed_point_index: int,
    example_tracked_crop_index: int | None,
    example_grid_crop_index: int | None,
    output_path: Path,
    figure_size: tuple[float, float] = (1.85, 1.95),
    minimum_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
    fixed_point_radius_threshold: float = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
    min_num_traj_per_bin: int = 10,
) -> Path:
    # Load and process trajectory dataframes for first passage time analysis and
    # visualization. Also load the fixed points dataframe, which will be used
    # for filtering to trajectories that reach the fixed point and for plotting
    # the fixed point in the 3D feature space.
    traj_df_grid, traj_df_tracked, fixed_points_df = _load_trajectory_dataframes(
        dataset_name=dataset_name,
        minimum_track_length=minimum_track_length,
    )

    # Bin (theta, r, rho) feature space.
    # First, define the bin sizes for each feature to be binned
    bin_sizes = {
        Column.DiffAEData.POLAR_ANGLE: np.pi / 12,  # 15 degree bins for the angle
        Column.DiffAEData.POLAR_RADIUS: 0.25,
        Column.DiffAEData.PC3_FLIPPED: 0.5,
    }

    # Then, get the data limits for each feature to be binned
    bin_limits: dict = {}
    for col in DYNAMICS_COLUMN_NAMES:
        col_min = min(traj_df_grid[col].min(), traj_df_tracked[col].min())
        col_max = max(traj_df_grid[col].max(), traj_df_tracked[col].max())
        bin_limits[col] = (col_min, col_max)

    # Finally, adjust the bin_limits if the feature has a defined range (e.g. for
    # angles, range is 0 to pi in radians)
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

    # Filter the trajectory dataframes to only include trajectories that reach
    # the stable fixed point, since those are the only trajectories for which
    # first passage times to the fixed point are defined.
    traj_df_grid_sub, traj_df_tracked_sub = _filter_to_first_passage_trajectories(
        traj_df_grid=traj_df_grid,
        traj_df_tracked=traj_df_tracked,
        fixed_point_index=example_fixed_point_index,
        fixed_point_radius_threshold=fixed_point_radius_threshold,
    )

    # Compute the first passage time for each trajectory
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

    fpt_stats_df = _compute_filtered_fpt_stats(
        traj_df_grid_sub=traj_df_grid_sub,
        traj_df_tracked_sub=traj_df_tracked_sub,
        dataset_name=dataset_name,
        fixed_point_index=example_fixed_point_index,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        min_num_traj_per_bin=min_num_traj_per_bin,
    )

    example_bin_center, example_bin_edges = _select_example_bin(
        fpt_stats_df=fpt_stats_df,
        min_num_traj_per_bin=min_num_traj_per_bin,
    )

    example_traj_df_grid, example_traj_df_tracked = _select_example_bin_trajectories(
        example_bin_center=example_bin_center,
        bin_edges=bin_edges,
        traj_df_grid_sub=traj_df_grid_sub,
        traj_df_tracked_sub=traj_df_tracked_sub,
        example_tracked_crop_index=example_tracked_crop_index,
        example_grid_crop_index=example_grid_crop_index,
    )

    example_bin_cuboid_edges = _build_bin_cuboid_edges(
        example_bin_edges=example_bin_edges,
        bin_sizes=bin_sizes,
    )

    fig = _create_fpt_schematic_figure(
        figure_size=figure_size,
        example_traj_df_tracked=example_traj_df_tracked,
        example_traj_df_grid=example_traj_df_grid,
        fixed_points_df=fixed_points_df,
        example_fixed_point_index=example_fixed_point_index,
        fixed_point_radius_threshold=fixed_point_radius_threshold,
        example_bin_cuboid_edges=example_bin_cuboid_edges,
        grid_bin_edges=bin_edges,
    )

    filename = f"{dataset_name}_FPT_fp_{example_fixed_point_index}_mean_3d_scatter"
    save_plot_to_path(fig, output_path, filename, file_format=".svg", bbox_inches="tight")

    return output_path / f"{filename}.svg"
