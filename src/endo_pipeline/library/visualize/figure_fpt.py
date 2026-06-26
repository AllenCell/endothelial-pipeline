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
from endo_pipeline.library.analyze.first_passage_time import (
    add_first_passage_time_column,
    build_first_passage_time_bins,
    compute_first_passage_time_statistics,
    filter_first_passage_time_by_min_num_trajectories,
    filter_to_trajectories_reaching_fixed_point,
    load_dataframes_for_first_passage_time_analysis,
)
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    LONG_TRACK_THRESHOLD_LENGTH,
    POLAR_ANGLE_PERIOD,
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


def _select_example_bin_trajectories(
    example_bin_center: np.ndarray,
    bin_edges: list[np.ndarray],
    traj_df_grid_sub: pd.DataFrame,
    traj_df_tracked_sub: pd.DataFrame,
    example_tracked_crop_index: int | None,
    example_grid_crop_index: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter dataframes to only include a single trajectory from each patch type
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
    # colored by patch type
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
    traj_df_grid, traj_df_tracked, fixed_points_df = (
        load_dataframes_for_first_passage_time_analysis(
            dataset_name=dataset_name,
            minimum_track_length=minimum_track_length,
        )
    )

    bin_edges, bin_centers, _, _ = build_first_passage_time_bins(
        traj_df_grid=traj_df_grid, traj_df_tracked=traj_df_tracked
    )

    # Filter the trajectory dataframes to only include trajectories that reach
    # the stable fixed point, since those are the only trajectories for which
    # first passage times to the fixed point are defined.
    traj_df_grid_sub, traj_df_tracked_sub = filter_to_trajectories_reaching_fixed_point(
        traj_df_grid=traj_df_grid,
        traj_df_tracked=traj_df_tracked,
        fixed_point_index=example_fixed_point_index,
        fixed_point_radius_threshold=fixed_point_radius_threshold,
    )

    # Compute the first passage time for each trajectory
    traj_df_grid_sub = add_first_passage_time_column(
        fixed_point_index=example_fixed_point_index,
        trajectory_df=traj_df_grid_sub,
        threshold=fixed_point_radius_threshold,
        time_column=Column.SegData.TIME_HRS,
    )
    traj_df_tracked_sub = add_first_passage_time_column(
        fixed_point_index=example_fixed_point_index,
        trajectory_df=traj_df_tracked_sub,
        threshold=fixed_point_radius_threshold,
        time_column=Column.SegData.TIME_HRS,
    )

    fpt_stats_df = compute_first_passage_time_statistics(
        traj_df_grid_sub=traj_df_grid_sub,
        traj_df_tracked_sub=traj_df_tracked_sub,
        dataset_name=dataset_name,
        fixed_point_index=example_fixed_point_index,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
    )

    # remove bins that don't have enough trajectories in them from either the
    # grid or tracked trajectories and drop nans
    fpt_stats_df_no_nan = filter_first_passage_time_by_min_num_trajectories(
        fpt_stats_df=fpt_stats_df,
        min_num_traj_per_bin=min_num_traj_per_bin,
        metric_for_filter="mean",
    )

    # Select an example bin to visualize trajectories
    example_bin_center, example_bin_edges = fpt_stats_df_no_nan[
        [Column.VectorField.BIN_CENTER, Column.VectorField.BIN_EDGES]
    ].iloc[0]

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
