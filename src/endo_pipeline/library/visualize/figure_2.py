"""Helper functions for visualizations used in Figure 2."""

from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Rectangle as MplRectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_vector_field_as_dict_from_dataframe,
    load_drift_dataframe_for_dataset,
)
from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    make_legend_handles_for_fixed_pts,
    plot_drift_1d,
    plot_drift_contours,
)
from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.figures import FONTSIZE_SMALL, FONTSIZE_XSMALL
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode


def _add_colorbar_to_contour_plot(
    fig: plt.Figure,
    axes: plt.Axes,
    vmin: float = DRIFT_CONTOUR_VMIN,
    vmax: float = DRIFT_CONTOUR_VMAX,
    ticks: np.ndarray | None = None,
    tick_label_round: int = DRIFT_CONTOUR_CBAR_ROUND,
    colormap: str = DRIFT_CONTOUR_COLORMAP,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    cax_position: Literal["top", "bottom", "left", "right"] = "right",
    extend: Literal["neither", "both", "min", "max"] = "both",
    pad: float = 0.03,
) -> None:
    """
    Add a colorbar to a contour plot with specified formatting.

    Parameters
    ----------
    fig
        Matplotlib figure object containing the contour plot.
    axes
        Matplotlib axes object containing the contour plot.
    vmin
        Minimum value for the colorbar.
    vmax
        Maximum value for the colorbar.
    ticks
        Array of tick values for the colorbar. If None, ticks will be generated
        automatically based on `vmin`, `vmax`, and
        `DRIFT_CONTOUR_CBAR_NUM_TICKS`.
    tick_label_round
        Number of decimal places to round colorbar tick labels to.
    colormap
        Colormap to use for the colorbar.
    orientation
        Orientation of the colorbar, either "vertical" or "horizontal".
    cax_position
        Position of the colorbar axes relative to the main axes, one of "top",
        "bottom", "left", or "right".
    pad
        Padding between the main axes and the colorbar axes, in inches.

    """
    cax = inset_axes(
        axes,
        width="100%" if orientation == "horizontal" else "5%",
        height="5%" if orientation == "horizontal" else "100%",
        loc="lower left",
        bbox_to_anchor=(
            (0, 1.0 + pad, 1, 1) if orientation == "horizontal" else (1.0 + pad, 0, 1, 1)
        ),
        bbox_transform=axes.transAxes,
        borderpad=0,
    )

    color_mappable = ScalarMappable(
        norm=TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0), cmap=colormap
    )
    colorbar_ticks = (
        ticks
        if ticks is not None
        else np.linspace(
            np.round(vmin, tick_label_round),
            np.round(vmax, tick_label_round),
            DRIFT_CONTOUR_CBAR_NUM_TICKS,
        )
    )
    colorbar_ticks = np.round(colorbar_ticks, tick_label_round)

    fig.colorbar(
        color_mappable, cax=cax, orientation=orientation, ticks=colorbar_ticks, extend=extend
    )

    tick_axis = cax.xaxis if orientation == "horizontal" else cax.yaxis
    tick_axis.set_ticks_position(cax_position)
    tick_axis.set_label_position(cax_position)


def _get_nullcline_coords_from_contour_axes(
    axes: np.ndarray[plt.Axes, Any],
    column_names: list[Column.DiffAEData],
    r_lims: tuple[float, float],
    rho_lims: tuple[float, float],
) -> dict[Column.DiffAEData, pd.DataFrame]:
    """
    Extract (r, rho) coordinates along nullclines from the contour collections
    in the given axes.

    Parameters
    ----------
    axes
        Axes objects containing the contour plots with nullclines.
    column_names
        List of column names corresponding to the nullclines, in the same order
        as they are plotted in the axes.
    r_lims
        Tuple specifying the limits of the r-axis, used to clip nullcline
        coordinates to the plotted range.
    rho_lims
        Tuple specifying the limits of the rho-axis, used to clip nullcline
        coordinates to the plotted range.

    Returns
    -------
    :
        Dictionary mapping each column name to a DataFrame containing the (r,
        rho) coordinates of the corresponding nullcline.

    """
    nullcline_coords = {col: pd.DataFrame() for col in column_names}
    for index, column in enumerate(column_names):
        # get coordinates from the contour collections corresponding to the
        # nullclines (these are generated by `plot_drift_contours` with
        # `include_nullclines=True`, last thing plotted on each subplot)
        nullcline_collection = axes[index].collections[-1]
        nullcline_paths = nullcline_collection.get_paths()
        for path in nullcline_paths:
            path_coords_ = path.vertices.T.copy()
            # clip to the axes limits to avoid including coordinates outside the
            # plotted range
            is_in_bounds = (
                (path_coords_[0] >= r_lims[0])
                & (path_coords_[0] <= r_lims[1])
                & (path_coords_[1] >= rho_lims[0])
                & (path_coords_[1] <= rho_lims[1])
            )
            path_coords = path_coords_[:, is_in_bounds]
            # sort so that coordinates are ordered by:
            #  - increasing rho for the r-nullcline (so that we can generate images by increasing rho)
            #  - increasing r for the rho-nullcline (so that we can generate images by increasing r)
            sort_arg = 1 if column == Column.DiffAEData.POLAR_RADIUS else 0
            sort_indices = np.argsort(path_coords[sort_arg])
            path_coords[0] = path_coords[0][sort_indices]
            path_coords[1] = path_coords[1][sort_indices]

            # append the coordinates to the DataFrame for the corresponding column
            nullcline_coords[column] = pd.concat(
                [
                    nullcline_coords[column],
                    pd.DataFrame(
                        {
                            Column.DiffAEData.POLAR_RADIUS: path_coords[0],
                            Column.DiffAEData.PC3_FLIPPED: path_coords[1],
                        }
                    ),
                ],
                ignore_index=True,
            )
    return nullcline_coords


def _get_example_points_along_nullcline(
    nullcline_coords: dict[Column.DiffAEData, pd.DataFrame],
    stable_fixed_point: np.ndarray,
    num_points: int = 5,
) -> dict[Column.DiffAEData, np.ndarray]:
    """
    Get coordinates of example points along each nullcline, selected to be
    approximately equally spaced by arc length along the nullcline.

    Parameters
    ----------
    nullcline_coords
        Dictionary mapping each column name to a DataFrame containing the (r,
        rho) coordinates of the corresponding nullcline.
    stable_fixed_point
        Coordinates of the stable fixed point, used as a reference point for
        selecting example points along the nullcline.
    num_points
        Number of example points to select along each nullcline.

    Returns
    -------
    :
        Dictionary mapping each column name to an array of shape (2, num_points)
        containing the (r, rho) coordinates of the selected example points
        along the corresponding nullcline.

    """
    example_points = {}
    for column, coords_dataframe in nullcline_coords.items():
        path_coords = (
            coords_dataframe[[Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]]
            .to_numpy()
            .T
        )

        # calculate arc length along the nullcline path, inserting a 0 at the
        # beginning for the first point
        arc_length = np.cumsum(np.sqrt(np.sum(np.diff(path_coords, axis=1) ** 2, axis=0)))
        arc_length = np.insert(arc_length, 0, 0)

        # starting with the fixed point as reference, find 4 additional points
        # (on either side of the fixed point if possible) along the nullcline
        # that are equally spaced by arc length
        fixed_point_index = np.argmin(
            np.sqrt(
                np.sum((path_coords - stable_fixed_point.flatten()[:, np.newaxis]) ** 2, axis=0)
            )
        )
        total_length = arc_length[-1]
        example_arc_lengths = np.linspace(
            max(0, arc_length[fixed_point_index] - total_length / 2),
            min(total_length, arc_length[fixed_point_index] + total_length / 2),
            num_points,
        )
        example_indices = [
            np.argmin(np.abs(arc_length - example_arc_length))
            for example_arc_length in example_arc_lengths
        ]
        # get coordinates of the selected example points
        example_points[column] = path_coords[:, example_indices]

        # replace the point closest to the stable fixed point with the stable
        # fixed point
        closest_index = np.argmin(
            np.sqrt(
                np.sum(
                    (example_points[column] - stable_fixed_point.flatten()[:, np.newaxis]) ** 2,
                    axis=0,
                )
            )
        )
        example_points[column][:, closest_index] = stable_fixed_point.flatten()

    return example_points


def make_2d_contour_plot_panel(
    drift: np.ndarray,
    meshgrid: tuple[np.ndarray, np.ndarray],
    column_labels: list[str],
    stable_fixed_point: np.ndarray,
    figsize: tuple[float, float],
    fig_savedir: Path,
    filename: str,
    r_lims: tuple[float, float],
    rho_lims: tuple[float, float],
    r_ticks: list[float],
    rho_ticks: list[float],
    nullcline_r_style: str,
    nullcline_rho_style: str,
    nullcline_opacity: float,
    gridspec_kwargs: dict | None,
    xlabel_kwargs: dict | None,
    ylabel_kwargs: dict | None,
    axes_title_kwargs: dict | None,
    include_colorbar: bool = True,
    include_legend: bool = True,
) -> tuple[Path, dict[Column.DiffAEData, np.ndarray]]:
    """
    Make and save plot of drift contours in (r, rho) space for a given dataset.
    """
    column_names = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    column_labels = [COLUMN_METADATA[column].label or str(column) for column in column_names]
    # plot drift contours and save
    fig, axes_ = plot_drift_contours(
        meshgrid=meshgrid,
        drift=drift,
        variable_labels=column_labels,
        figsize=figsize,
        n_rows=1,
        n_cols=2,
        axes_limits=[r_lims, rho_lims],
        axes_aspect=None,
        axes_titles=(f"d{column_labels[0]}/dt", f"d{column_labels[1]}/dt"),
        include_colorbar=False,
        include_nullclines=True,
        nullcline_colors=("k", "k"),
        nullcline_styles=(nullcline_r_style, nullcline_rho_style),
        nullcline_opacity=nullcline_opacity,
        gridspec_kwargs=gridspec_kwargs,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        axes_title_kwargs=axes_title_kwargs,
    )
    # get (r, rho) coordinates of r- and rho-nullclines for generating images
    axes = cast(np.ndarray[plt.Axes, Any], axes_)
    nullcline_coords_ = _get_nullcline_coords_from_contour_axes(
        axes, column_names, r_lims, rho_lims
    )

    # get 5 equally space (by arc length) coordinates along the nullcline to
    # plot as points on top of the contour, including the stable fixed point and
    # 2 points on either side if possible
    nullcline_coords = _get_example_points_along_nullcline(nullcline_coords_, stable_fixed_point)

    for ax_index, ax_ in enumerate(list(axes_)):
        # add nullcline points on top of the contour plot
        ax_.plot(
            nullcline_coords[column_names[ax_index]][0],
            nullcline_coords[column_names[ax_index]][1],
            "o",
            color="w",
            markeredgecolor="k",
            markeredgewidth=0.25,
            markersize=3,
        )
        # add stable fixed point on top of the contour plot
        ax_.plot(
            stable_fixed_point[..., 0],
            stable_fixed_point[..., 1],
            FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
            color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
            markeredgecolor="k",
            markeredgewidth=0.5,
            markersize=5,
        )
        # adjust label padding and drop tick labels on shared y axis
        ax_.set_box_aspect(1.0)
        ax_.set_xticks(r_ticks)
        ax_.set_yticks(rho_ticks)
        if ax_index == 1:
            ax_.tick_params(labelleft=False)

    # if indicated, add colorbar to the top of the first subplot with ticks and
    # label formatting
    if include_colorbar:
        _add_colorbar_to_contour_plot(fig, axes_[1])
        # shrink the constrained-layout region so the inset colorbar axes
        # (which lives outside the main axes boundary) is not clipped on save

        layout_engine = fig.get_layout_engine()
        if isinstance(layout_engine, ConstrainedLayoutEngine):
            layout_engine.set(rect=(0, 0, 0.9, 1))

    handles = []
    labels = []
    if include_legend:
        # plot_drift_contours draws nullclines via ax.contour(), which does not
        # produce labeled artists. Add proxy Line2D handles so the legend has
        # something to show.
        nullcline_styles = (nullcline_r_style, nullcline_rho_style)
        for col_idx, col in enumerate(column_names):
            label = COLUMN_METADATA[col].label or str(col)
            handle = mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=nullcline_styles[col_idx],
                label=f"Nullcline d{label}/dt=0",
            )
            handles.append(handle)
            labels.append(f"Nullcline d{label}/dt=0")
        fig.legend(
            handles,
            labels,
            fontsize="xx-small",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=2,
            handletextpad=0.3,
        )

    save_plot_to_path(
        fig,
        fig_savedir,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        pad_inches=0,
    )

    return fig_savedir / f"{filename}.svg", nullcline_coords


def make_1d_drift_plot_panel(
    drift: np.ndarray,
    theta_values: np.ndarray,
    column_label: str,
    stable_fixed_point: float,
    figsize: tuple[float, float],
    fig_savedir: Path,
    filename: str,
    axes_xlim: tuple[float, float],
    axes_ylim: tuple[float, float],
    axes_xticks: list[float],
    axes_xtick_labels: list[str],
    axes_yticks: list[float],
    arrow_scale: float,
    arrow_width: float,
    drift_line_kwargs: dict | None,
    zero_line_kwargs: dict | None,
    gridspec_kwargs: dict | None,
    xlabel_kwargs: dict | None,
    ylabel_kwargs: dict | None,
) -> Path:
    fig, ax = plot_drift_1d(
        drift=drift,
        x_values=theta_values,
        figsize=figsize,
        axes_limits=[axes_xlim, axes_ylim],
        axes_labels=[column_label, f"d{column_label}/dt"],
        add_flow_arrows=True,
        flow_arrow_kwargs={"color": "dimgrey", "scale": arrow_scale, "width": arrow_width},
        flow_arrow_downsample=10,
        gridspec_kwargs=gridspec_kwargs,
        drift_line_kwargs=drift_line_kwargs,
        zero_line_kwargs=zero_line_kwargs,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )
    # add stable fixed point in theta
    ax.plot(
        stable_fixed_point,
        np.zeros_like(stable_fixed_point),
        FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
        color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
        markeredgecolor="k",
        markeredgewidth=0.5,
        markersize=5,
    )

    # set plot formatting args
    ax.set_box_aspect(1.0)
    ax.set_xticks(axes_xticks, labels=axes_xtick_labels)
    ax.set_yticks(axes_yticks)

    save_plot_to_path(
        fig, fig_savedir, filename, file_format=".svg", tight_layout=False, transparent=True
    )

    return fig_savedir / f"{filename}.svg"


def reconstruct_along_nullcline(
    nullcline_coords: dict[ColumnNameType, np.ndarray],
    theta_value: float,
    model: DiffusionAutoEncoder,
    fig_savedir: Path,
    num_gpus: int | None = None,
    random_seed: int | None = 4,
) -> Path:
    """
    Generate reconstructed images along a nullcline given the coordinates of the
    nullcline in feature space.

    Parameters
    ----------
    nullcline_coords
        Dictionary containing DataFrames with the coordinates of each nullcline.
    theta_value
        Value of the theta feature to use for generating the images along the
        nullcline (since the nullcline coordinates only contain r and rho).
    model
        DiffusionAutoEncoder model used to generate synthetic images.
    fig_savedir
        Directory where the generated figures will be saved.
    num_gpus
        Number of GPUs to use for image generation. If None, will use all
        available GPUs.
    random_seed
        Random seed for reproducibility of image generation. If None, will use a
        random seed.
    """
    column_names = cast(list[str], list(DYNAMICS_COLUMN_NAMES))
    coords_dataframes_: list[pd.DataFrame] = []
    for _, coords_dataframe in nullcline_coords.items():
        r_coords = coords_dataframe[0]
        rho_coords = coords_dataframe[1]
        full_coords_dataframe = pd.DataFrame(  # add theta values
            {
                Column.DiffAEData.POLAR_ANGLE: theta_value * np.ones_like(r_coords),
                Column.DiffAEData.POLAR_RADIUS: r_coords,
                Column.DiffAEData.PC3_FLIPPED: rho_coords,
            }
        )
        coords_dataframes_.append(full_coords_dataframe)
    coords_dataframes = pd.concat(coords_dataframes_, ignore_index=True)

    # reconstruct images along the nullcline coordinates and make a contact
    # sheet of the results
    walk_array = generate_from_dataframe(
        coords_dataframes,
        column_names,
        model,
        num_gpus=num_gpus,
        random_seed=random_seed,
    )
    walk_panels = [walk_array[i] for i in range(len(walk_array))]

    n_cols = len(walk_array) // 2
    fig_null_walks = make_contact_sheet(
        panels=walk_panels,
        max_rows=2,
        max_cols=n_cols,
        fig_kwargs={"figsize": (3.125, 1.3), "layout": "constrained"},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
    )

    # add a single outline box spanning both rows of the center column (stable
    # fixed point) in the stable fixed point color
    stable_color = FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color
    center_col = n_cols // 2
    # trigger constrained layout so ax.get_position() reflects final geometry
    fig_null_walks.canvas.draw()
    ax_top = fig_null_walks.axes[0 * n_cols + center_col]
    ax_bot = fig_null_walks.axes[1 * n_cols + center_col]
    pos_top = ax_top.get_position()
    pos_bot = ax_bot.get_position()
    # combine: left/width from top (same column), y from bottom of lower axes,
    # height spans from bottom of lower axes to top of upper axes; add a small
    # padding (in figure-fraction units) so the box doesn't sit on the axes edge
    pad = 0.0125
    rect = MplRectangle(
        (pos_top.x0 - pad, pos_bot.y0 - 2 * pad),
        pos_top.width + 2 * pad,
        pos_top.y1 - pos_bot.y0 + 4 * pad,
        linewidth=1.5,
        edgecolor=stable_color,
        facecolor="none",
        transform=fig_null_walks.transFigure,
        clip_on=False,
    )
    fig_null_walks.add_artist(rect)

    # add row labels to the right of each nullcline row
    ax_top_right = fig_null_walks.axes[n_cols - 1]
    ax_bot_right = fig_null_walks.axes[2 * n_cols - 1]
    pos_top_right = ax_top_right.get_position()
    pos_bot_right = ax_bot_right.get_position()
    label_x = pos_top_right.x1 + 0.025
    for pos, label in [
        (pos_top_right, "r-nullcline"),
        (pos_bot_right, f"{Unicode.RHO}-nullcline"),
    ]:
        fig_null_walks.text(
            label_x,
            pos.y0 + pos.height / 2,
            label,
            va="center",
            ha="left",
            fontsize=FONTSIZE_SMALL,
            fontweight="bold",
        )

    filename = "nullcline_walks"
    save_plot_to_path(
        fig_null_walks,
        fig_savedir,
        filename,
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )

    return fig_savedir / f"{filename}.svg"


def _load_and_process_vector_field(
    dataset_name: str,
    column_names: list[Column.DiffAEData],
    theta_lims: tuple[float, float],
    r_lims: tuple[float, float],
    rho_lims: tuple[float, float],
    downsample_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the vector field data for the given dataset and process it to
    extract the grid and vector components within the specified limits,
    returning downsampled versions for plotting.
    """
    drift_df = load_drift_dataframe_for_dataset(dataset_name)
    flow_field_dict = get_vector_field_as_dict_from_dataframe(drift_df, column_names)

    # grids and vectors are 3-D arrays shaped (n_theta, n_r, n_rho)
    x_grid_, y_grid_, z_grid_ = flow_field_dict["grid"]
    u_field_, v_field_, w_field_ = flow_field_dict["vectors"]

    # mask vector field to be within the specified limits (take only grid points
    # within limits), reshaping accordingly as 3D arrays of updated number of
    # points within limits
    x_in_bounds = (x_grid_ >= theta_lims[0]) & (x_grid_ <= theta_lims[1])
    num_x_in_bounds = np.unique(np.sum(x_in_bounds, axis=0))[-1]
    y_in_bounds = (y_grid_ >= r_lims[0]) & (y_grid_ <= r_lims[1])
    num_y_in_bounds = np.unique(np.sum(y_in_bounds, axis=1))[-1]
    z_in_bounds = (z_grid_ >= rho_lims[0]) & (z_grid_ <= rho_lims[1])
    num_z_in_bounds = np.unique(np.sum(z_in_bounds, axis=2))[-1]
    in_bounds_mask = x_in_bounds & y_in_bounds & z_in_bounds

    x_grid = x_grid_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    y_grid = y_grid_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    z_grid = z_grid_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    u_field = u_field_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    v_field = v_field_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    w_field = w_field_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)

    # Downsample uniformly along every axis
    x_ds = x_grid[::downsample_factor, ::downsample_factor, ::downsample_factor]
    y_ds = y_grid[::downsample_factor, ::downsample_factor, ::downsample_factor]
    z_ds = z_grid[::downsample_factor, ::downsample_factor, ::downsample_factor]
    u_ds = u_field[::downsample_factor, ::downsample_factor, ::downsample_factor]
    v_ds = v_field[::downsample_factor, ::downsample_factor, ::downsample_factor]
    w_ds = w_field[::downsample_factor, ::downsample_factor, ::downsample_factor]

    return x_ds, y_ds, z_ds, u_ds, v_ds, w_ds


def _plot_quiver_3d_cones(
    ax: Axes3D,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    length: float,
    colors: np.ndarray,
    alpha: float = 0.8,
    cone_fraction: float = 0.30,
    cone_radius_ratio: float = 0.4,
    n_facets: int = 8,
) -> None:
    """
    Draw 3-D quiver arrows with cone-shaped arrowheads using
    :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection` for the shafts and
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection` for the cone faces.

    Unlike :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.quiver`, every arrowhead
    is a proper closed cone so arrows look volumetric from any viewing angle.

    Each arrow is decomposed into:

        - **shaft**: a single line segment from the tail to the cone base.
        - **cone side faces**: ``n_facets`` triangles between the base circle
          and the apex.
        - **cone base disc**: ``n_facets`` triangles that cap the open base of
          the cone so it appears solid when viewed from behind.

    All geometry is batched into two
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection` objects (sides and
    base) for efficient rendering.

    Parameters
    ----------
    ax
        The 3-D axes on which to draw.
    x, y, z
        Flat arrays of arrow tail positions.
    u, v, w
        Flat arrays of arrow direction components.  They are normalised
        internally, so only the direction matters; overall arrow length is
        controlled by ``length``.
    length
        Total arrow length in data units (shaft + cone).
    colors
        RGBA colour array of shape ``(N, 4)`` — one colour per arrow.
    alpha
        Overall opacity applied to both shaft lines and cone faces.
    cone_fraction
        Fraction of ``length`` occupied by the cone head.  The shaft fills the
        remaining ``1 - cone_fraction`` portion.
    cone_radius_ratio
        Cone base radius expressed as a fraction of the cone height.  Larger
        values produce stubbier, more visible heads.
    n_facets
        Number of triangular side faces on each cone.  8 gives a smooth
        appearance without excessive vertex count.

    """
    eps = np.finfo(float).eps
    mag = np.sqrt(u**2 + v**2 + w**2)

    # unit direction vectors; fall back to z-axis for zero-magnitude vectors
    with np.errstate(invalid="ignore", divide="ignore"):
        ud = np.where(mag > eps, u / mag, 0.0)
        vd = np.where(mag > eps, v / mag, 0.0)
        wd = np.where(mag > eps, w / mag, np.ones_like(w))

    shaft_length = length * (1.0 - cone_fraction)
    cone_height = length * cone_fraction
    cone_radius = cone_height * cone_radius_ratio

    # tip (apex) and cone-base centre for every arrow
    tip_x = x + ud * length
    tip_y = y + vd * length
    tip_z = z + wd * length
    base_x = x + ud * shaft_length
    base_y = y + vd * shaft_length
    base_z = z + wd * shaft_length

    # ------------------------------------------------------------------ shafts
    shaft_segs = np.stack(
        [
            np.column_stack([x, y, z]),
            np.column_stack([base_x, base_y, base_z]),
        ],
        axis=1,
    )  # (N, 2, 3)
    shaft_col = Line3DCollection(shaft_segs, colors=colors, alpha=alpha, linewidths=0.8)
    ax.add_collection3d(shaft_col)

    # ------------------------------------------------------------------ cones
    # Build two orthonormal vectors perpendicular to each direction
    # to parameterise the base circle.
    arbitrary = np.where(
        (np.abs(wd) < 0.9)[:, np.newaxis],
        np.tile([0.0, 0.0, 1.0], (len(ud), 1)),
        np.tile([1.0, 0.0, 0.0], (len(ud), 1)),
    )  # (N, 3)
    d_vec = np.column_stack([ud, vd, wd])  # (N, 3)
    perp1 = np.cross(d_vec, arbitrary)
    perp1 /= np.linalg.norm(perp1, axis=1, keepdims=True) + eps
    perp2 = np.cross(d_vec, perp1)
    perp2 /= np.linalg.norm(perp2, axis=1, keepdims=True) + eps

    # angles for base-circle vertices
    angles = np.linspace(0, 2 * np.pi, n_facets, endpoint=False)
    cos_a = np.cos(angles)  # (n_facets,)
    sin_a = np.sin(angles)  # (n_facets,)

    # base circle points: shape (N, n_facets, 3)
    circle = (
        np.array([base_x, base_y, base_z]).T[:, np.newaxis, :]  # (N,1,3)
        + cone_radius * cos_a[np.newaxis, :, np.newaxis] * perp1[:, np.newaxis, :]
        + cone_radius * sin_a[np.newaxis, :, np.newaxis] * perp2[:, np.newaxis, :]
    )

    apex = np.column_stack([tip_x, tip_y, tip_z])  # (N, 3)

    # side faces: triangles (apex, circle[i], circle[i+1])
    next_i = (np.arange(n_facets) + 1) % n_facets
    side_verts = np.stack(
        [
            apex[:, np.newaxis, :].repeat(n_facets, axis=1),  # (N, n_facets, 3)
            circle,  # (N, n_facets, 3)
            circle[:, next_i, :],  # (N, n_facets, 3)
        ],
        axis=2,
    )  # (N, n_facets, 3-verts, 3-coords)
    N = len(x)
    side_verts_list = side_verts.reshape(N * n_facets, 3, 3).tolist()
    side_colors = np.repeat(colors, n_facets, axis=0)

    side_col = Poly3DCollection(
        side_verts_list,
        facecolors=side_colors,
        edgecolors="none",
        alpha=alpha,
    )
    ax.add_collection3d(side_col)

    # base disc: triangles (base_centre, circle[i], circle[i+1])
    base_centre = np.column_stack([base_x, base_y, base_z])
    base_verts = np.stack(
        [
            base_centre[:, np.newaxis, :].repeat(n_facets, axis=1),
            circle,
            circle[:, next_i, :],
        ],
        axis=2,
    )  # (N, n_facets, 3, 3)
    base_verts_list = base_verts.reshape(N * n_facets, 3, 3).tolist()

    base_col = Poly3DCollection(
        base_verts_list,
        facecolors=side_colors,
        edgecolors="none",
        alpha=alpha,
    )
    ax.add_collection3d(base_col)


def _set_3d_axes_ticks_and_labels(
    ax: plt.Axes,
    x_label: str,
    x_ticks: list[float],
    x_tick_labels: list[str],
    y_label: str,
    y_ticks: list[float],
    y_tick_labels: list[str],
    z_label: str,
    z_ticks: list[float],
    z_tick_labels: list[str],
) -> None:
    ax.tick_params(axis="both", pad=-3)
    ax.set_xlabel(x_label, labelpad=-8)
    ax.set_xticks(x_ticks, labels=x_tick_labels)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_ha("right")
        tick.set_va("center")
    ax.set_ylabel(y_label, labelpad=-5)
    ax.set_yticks(y_ticks, labels=y_tick_labels)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_ha("left")
        tick.set_va("center")
    ax.set_zlabel(z_label, labelpad=-8)
    ax.set_zticks(z_ticks, labels=z_tick_labels)
    ax.zaxis.set_rotate_label(False)


def make_3d_vector_field_plot_panel(
    dataset_name: str,
    fig_savedir: Path,
    downsample_factor: int = 6,
    colormap: str = "viridis_r",
    magnitude_limits: tuple[float, float] = (5e-2, 1.5),
    arrow_alpha: float = 0.6,
) -> Path:
    """
    Render the 3D (theta, r, rho) drift vector field for a given dataset using
    matplotlib, with the stable fixed point overlaid as a scatter marker.

    The drift vector field is loaded via
    :func:`~endo_pipeline.library.analyze.vector_field_estimation.load_drift_dataframe_for_dataset`
    and the stable fixed point is loaded from the bootstrapped fixed-point
    dataframe manifest (``bootstrapped_fixed_points_grid``).

    Parameters
    ----------
    dataset_name
        Name of the dataset to visualize.
    fig_savedir
        Directory in which to save the figure as a static PNG file.
    downsample_factor
        Factor by which to downsample the grid before plotting, to keep the
        figure responsive.
    colormap
        Matplotlib-compatible colormap name used to colour the arrows by vector
        magnitude.
    magnitude_limits
        Tuple specifying the minimum and maximum magnitude values for the colour
        scale.  ``None`` uses the true minimum and maximum.
    arrow_alpha
        Opacity of the quiver arrows (0 = fully transparent, 1 = fully opaque).
        Lowering this value lets the stable fixed point marker show through the
        arrow field.

    Returns
    -------
    :
        Path to the saved figure file.

    """

    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    col_labels = [(COLUMN_METADATA[col].label or str(col)) for col in DYNAMICS_COLUMN_NAMES]

    figsize = (2.0, 2.5)
    theta_lims = (0, np.pi)
    theta_ticks = [0, np.pi / 2, np.pi]
    theta_tick_labels = [f"0={Unicode.PI}", f"{Unicode.PI}/2", f"{Unicode.PI}=0"]
    r_lims = (0, 1.75)
    r_ticks = [0.25, 0.75, 1.25]
    rho_lims = (-1.5, 1.5)
    rho_ticks = [-1.0, 0, 1.0]

    # Load, clip, and downsample drift vector field
    x_ds, y_ds, z_ds, u_ds, v_ds, w_ds = _load_and_process_vector_field(
        dataset_name,
        column_names,
        theta_lims,
        r_lims,
        rho_lims,
        downsample_factor,
    )

    # Compute vector magnitudes for colouring before normalizing to unit vectors
    # for plotting
    x_flat = x_ds.ravel()
    y_flat = y_ds.ravel()
    z_flat = z_ds.ravel()
    u_flat = u_ds.ravel()
    v_flat = v_ds.ravel()
    w_flat = w_ds.ravel()
    mag_flat = np.sqrt(u_flat**2 + v_flat**2 + w_flat**2)

    # Map magnitudes to colours
    mag_eps = 1e-10
    cmap = plt.get_cmap(colormap)
    safe_cmin = max(magnitude_limits[0], mag_eps)
    safe_cmax = max(magnitude_limits[1], safe_cmin + mag_eps)
    norm_log = LogNorm(vmin=safe_cmin, vmax=safe_cmax)
    colors = cmap(norm_log(np.clip(mag_flat, safe_cmin, safe_cmax)))
    scalar_mappable = ScalarMappable(cmap=cmap, norm=norm_log)

    # Build matplotlib 3D figure
    fig = plt.figure(figsize=figsize)
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    figsize_ratio = figsize[1] / figsize[0]
    ax.set_box_aspect((1.1 * figsize_ratio, 0.98 * figsize_ratio, 1.05 * figsize_ratio))
    ax.set_xlim(theta_lims)
    ax.set_ylim(r_lims)
    ax.set_zlim(rho_lims)

    # Render all arrows at the same absolute size (so visual clutter from
    # large-magnitude outliers is reduced) while still colouring by magnitude.
    avg_spacing = np.mean(np.diff(np.unique(x_flat)))
    arrow_length = avg_spacing * 0.8
    u_plot = u_flat / (mag_flat + mag_eps)
    v_plot = v_flat / (mag_flat + mag_eps)
    w_plot = w_flat / (mag_flat + mag_eps)
    _plot_quiver_3d_cones(
        ax,
        x_flat,
        y_flat,
        z_flat,
        u_plot,
        v_plot,
        w_plot,
        length=arrow_length,
        colors=colors,
        alpha=arrow_alpha,
    )

    # Colorbar - horizontal strip at the top, shifted left to leave room for legend
    scalar_mappable.set_array([])
    cbar_ax = fig.add_axes((0.1, 0.87, 0.48, 0.04))
    cbar = fig.colorbar(
        scalar_mappable,
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.ax.tick_params(labelsize=FONTSIZE_XSMALL, pad=2)
    cbar.set_label("$\Vert\mathbf{f}(\mathbf{x})\Vert$", fontsize=FONTSIZE_SMALL, labelpad=-1)
    cbar_ax.xaxis.set_label_position("top")
    cbar_ax.xaxis.tick_top()

    # Legend to the right of the colorbar
    arrow_handle = mlines.Line2D(
        [],
        [],
        color="gray",
        marker="$\\rightarrow$",
        markersize=8,
        linewidth=0.0,
        markevery=[0],
        label="$\mathbf{f}(\mathbf{x})$",
    )
    fp_handles = make_legend_handles_for_fixed_pts(
        fpt_stabilities=[StabilityLabel.STABLE],
        marker_size=4,
    )
    fig.legend(
        handles=[arrow_handle, *fp_handles],
        fontsize=FONTSIZE_XSMALL,
        loc="upper left",
        bbox_to_anchor=(0.65, 1.0),
        frameon=False,
        handletextpad=0.3,
        labelspacing=0.4,
        handler_map={arrow_handle: HandlerLine2D(numpoints=1)},
    )

    # Load and overlay stable fixed point
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)

    stable_df = fixed_points_df[
        fixed_points_df[Column.VectorField.STABILITY] == StabilityLabel.STABLE
    ]
    fpt_coords = stable_df[column_names].to_numpy()
    hex_color: str = FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color
    ax.scatter(
        fpt_coords[:, 0],
        fpt_coords[:, 1],
        fpt_coords[:, 2],
        color=hex_color,
        s=80,
        zorder=5,
    )

    # set axes labels and ticks with custom formatting
    ax.tick_params(axis="both", pad=-3)
    ax.set_xlabel(col_labels[0], labelpad=-8)
    ax.set_xticks(theta_ticks, labels=theta_tick_labels)
    ax.set_xlim(theta_lims)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_ha("right")
        tick.set_va("center")
    ax.set_ylabel(col_labels[1], labelpad=-5)
    ax.set_yticks(r_ticks)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_ha("left")
        tick.set_va("center")
    ax.set_ylim(r_lims)
    ax.set_zlabel(col_labels[2], labelpad=-8)
    ax.set_zticks(rho_ticks)
    ax.set_zlim(rho_lims)
    ax.zaxis.set_rotate_label(False)

    # save as .svg file
    filename = f"3d_vector_field_{dataset_name}"
    save_plot_to_path(
        fig,
        fig_savedir,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=False,
        bbox_inches="tight",
    )

    return fig_savedir / f"{filename}.svg"
