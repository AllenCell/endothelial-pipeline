"""Helper functions for visualizations used in Figure 2."""

from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import Rectangle as MplRectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.track_integration import get_line_fit_and_filtered_df
from endo_pipeline.library.analyze.vector_field_estimation import load_drift_dataframe_for_dataset
from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    plot_drift_1d,
    plot_drift_3d,
    plot_drift_contours,
    process_3d_vector_field_for_visualization,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.figures import FONTSIZE_SMALL, FONTSIZE_XSMALL
from endo_pipeline.settings.first_passage_time import FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.flow_field_figure import (
    AXES_LIMITS_2D,
    GRIDSPEC_KWARGS,
    NULLCLINE_STYLES_2D,
    XLABEL_KWARGS,
    YLABEL_KWARGS,
)
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME


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


@figure_panel("Make panel of 2D contour plots of drift in (r, rho) space.")
def make_2d_contour_plot_panel(
    figure_size: tuple[float, float],
    output_path: Path,
    drift: np.ndarray,
    meshgrid: tuple[np.ndarray, np.ndarray],
    column_labels: list[str],
    stable_fixed_point: np.ndarray,
    filename: str,
    include_legend: bool = True,
) -> tuple[Path, dict[Column.DiffAEData, np.ndarray]]:
    """
    Make and save plot of drift contours in (r, rho) space for a given dataset.
    """
    column_names = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    column_labels = [COLUMN_METADATA[column].label or str(column) for column in column_names]

    r_lims = AXES_LIMITS_2D[Column.DiffAEData.POLAR_RADIUS]
    rho_lims = AXES_LIMITS_2D[Column.DiffAEData.PC3_FLIPPED]
    r_ticks = [0.4, 1.0, 1.6]
    rho_ticks = [-0.75, 0.0, 0.75]
    nullcline_r_style = NULLCLINE_STYLES_2D[Column.DiffAEData.POLAR_RADIUS]
    nullcline_rho_style = NULLCLINE_STYLES_2D[Column.DiffAEData.PC3_FLIPPED]
    nullcline_opacity = 1.0
    gridspec_kwargs = GRIDSPEC_KWARGS
    xlabel_kwargs = XLABEL_KWARGS
    ylabel_kwargs = {**YLABEL_KWARGS, "rotation": 0}
    axes_title_kwargs = {
        "fontsize": FONTSIZE_SMALL,
        "x": 0.05,
        "y": 0.775,
        "rotation": 0,
        "ha": "left",
        "va": "center",
        "bbox": {
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    }

    # plot drift contours and save
    fig, axes_ = plot_drift_contours(
        meshgrid=meshgrid,
        drift=drift,
        variable_labels=column_labels,
        figsize=figure_size,
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

    # add colorbar to the right of the second subplot with ticks and label
    # formatting
    _add_colorbar_to_contour_plot(fig, axes_[1])

    # shrink the constrained-layout region so the inset colorbar axes
    # (which lives outside the main axes boundary) is not clipped on save
    layout_engine = fig.get_layout_engine()
    if isinstance(layout_engine, ConstrainedLayoutEngine):
        layout_engine.set(rect=(0, 0, 0.9, 1))

    if include_legend:
        handles = []
        labels = []
        # plot_drift_contours draws nullclines via ax.contour(), which does not
        # produce labeled artists. Add proxy Line2D handles so the legend has
        # something to show.
        nullcline_styles = (nullcline_r_style, nullcline_rho_style)
        for col_idx, col in enumerate(column_names):
            label = COLUMN_METADATA[col].label or str(col)
            legend_label = f"{label}-nullcline (d{label}/dt=0)"
            handle = mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=nullcline_styles[col_idx],
                label=legend_label,
            )
            handles.append(handle)
            labels.append(legend_label)
        fig.legend(
            handles,
            labels,
            fontsize="xx-small",
            loc="upper center",
            bbox_to_anchor=(0.525, 0.925),
            ncol=2,
            handletextpad=0.3,
        )

    save_plot_to_path(
        fig,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        pad_inches=0,
    )

    return output_path / f"{filename}.svg", nullcline_coords


@figure_panel("Make panel of 1D phase line plot of drift in theta space.")
def make_1d_drift_plot_panel(
    figure_size: tuple[float, float],
    output_path: Path,
    shear_stress_label: str,
    drift: np.ndarray,
    theta_values: np.ndarray,
    column_label: str,
    stable_fixed_point: float,
    filename: str,
    arrow_scale: float,
    arrow_width: float,
    include_legend: bool = True,
) -> Path:
    """Make and save plot of 1D drift as a function of theta for a given dataset."""
    axes_xlim = VECTOR_FIELD_THETA_RANGE
    axes_ylim = (-0.4, 0.4)

    # re-wrap theta values to be within the specified x-axis limits for better
    # visualization of the drift as a function of theta
    where_theta_below_xlim = theta_values < axes_xlim[0]
    where_theta_above_xlim = theta_values > axes_xlim[1]
    theta_values_wrapped = theta_values.copy()
    theta_values_wrapped[where_theta_below_xlim] += np.pi
    theta_values_wrapped[where_theta_above_xlim] -= np.pi
    arg_sorted_theta = np.argsort(theta_values_wrapped)
    theta_values_sorted = theta_values_wrapped[arg_sorted_theta]
    drift_sorted = drift[arg_sorted_theta]

    fig, ax = plot_drift_1d(
        drift=drift_sorted,
        x_values=theta_values_sorted,
        figsize=figure_size,
        axes_limits=[axes_xlim, axes_ylim],
        axes_labels=[column_label, shear_stress_label],
        add_flow_arrows=True,
        flow_arrow_kwargs={"color": "dimgrey", "scale": arrow_scale, "width": arrow_width},
        flow_arrow_downsample=10,
        gridspec_kwargs=GRIDSPEC_KWARGS,
        drift_line_kwargs={"color": "k", "linewidth": 2, "label": f"d{column_label}/dt"},
        zero_line_kwargs={
            "linestyle": "--",
            "color": "gray",
            "linewidth": 1,
            "alpha": 0.7,
            "label": f"d{column_label}/dt = 0",
        },
        xlabel_kwargs=XLABEL_KWARGS,
        ylabel_kwargs=XLABEL_KWARGS,
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

    # add legend
    if include_legend:
        fig.legend(
            fontsize="xx-small",
            loc="upper center",
            bbox_to_anchor=(0.65, 1.05),
            ncol=2,
            handletextpad=0.3,
            columnspacing=0.75,
        )

    # set plot formatting args
    ax.set_box_aspect(1.0)
    ax.set_xticks([0, np.pi / 2], labels=[f"0={Unicode.PI}", f"{Unicode.PI}/2"])
    ax.set_yticks([-0.3, 0.0, 0.3])

    save_plot_to_path(
        fig, output_path, filename, file_format=".svg", tight_layout=False, transparent=True
    )

    return output_path / f"{filename}.svg"


@figure_panel(
    "Reconstruct images at points along the given r- and rho-nullclines and save images as a contact sheet."
)
def reconstruct_along_nullcline(
    figure_size: tuple[float, float],
    output_path: Path,
    nullcline_coords: dict[ColumnNameType, np.ndarray],
    theta_value: float,
    model: DiffusionAutoEncoder,
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
    output_path
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
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
    )

    # add scalebars to each panel, only label the top left one to avoid
    # redundancy
    for i, ax in enumerate(fig_null_walks.axes):
        add_scalebar(
            ax,
            scale_bar_um=20,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            bar_thickness=5,
            padding=5,
            include_label=True if i == 0 else False,
            label_fontsize=FONTSIZE_XSMALL,
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

    # Add text above box with ({feat_1}^*, {feat_2}^*, {feat_3}^*) labeling the
    # fixed point, using the same stable color. Use set_in_layout(False) so that
    # constrained layout does not resize the axes to accommodate the text,
    # which would shift the box position
    above_box_txt = fig_null_walks.text(
        pos_top.x0 + pos_top.width / 2,
        pos_top.y1 + 2 * pad + 0.01,
        f"({Unicode.THETA}$^*$, r$^*$, {Unicode.RHO}$^*$)",
        color=stable_color,
        fontsize=FONTSIZE_XSMALL,
        fontweight="bold",
        ha="center",
        va="bottom",
        transform=fig_null_walks.transFigure,
        clip_on=False,
    )
    above_box_txt.set_in_layout(False)

    # add row labels to the left of each nullcline row
    ax_top_left = fig_null_walks.axes[0]
    ax_bot_left = fig_null_walks.axes[n_cols]
    pos_top_left = ax_top_left.get_position()
    pos_bot_left = ax_bot_left.get_position()
    label_x = pos_top_left.x0 - 0.15
    for pos, label in [
        (pos_top_left, "dr/dt=0"),
        (pos_bot_left, f"d{Unicode.RHO}/dt=0"),
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
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )

    return output_path / f"{filename}.svg"


@figure_panel("Reconstruct at fixed point.")
def reconstruct_fixed_points(
    fixed_point_df: pd.DataFrame,
    model: DiffusionAutoEncoder,
    figure_size: tuple[float, float],
    output_path: Path,
    num_gpus: int | None = None,
    random_seed: int | None = 4,
) -> Path:
    """
    Reconstruct the fixed point coordinates from the polar angle, radius, and
    rho columns.

    Parameters
    ----------
    fixed_point_df
        DataFrame containing the fixed point coordinates, with columns for polar
        angle, polar radius, and flipped PC3 (rho).
    model
        The diffusion autoencoder model used for reconstruction.
    fig_savedir
        Directory to save the reconstructed figures.
    num_gpus
        Number of GPUs to use for reconstruction. If None, will use CPU.
    random_seed_start
        Starting random seed for reproducibility.
    num_examples
        Number of examples to generate for each fixed point coordinate (by varying
        the random seed).
    """

    # reconstruct images along at the fixed point coordinates and make a contact
    # sheet of the results
    column_names = cast(list[str], list(DYNAMICS_COLUMN_NAMES))
    reconstructed_image = generate_from_dataframe(
        fixed_point_df,
        column_names,
        model,
        num_gpus=num_gpus,
        random_seed=random_seed,
    )

    fig_fixed_point_reconstructions = make_contact_sheet(
        panels=[reconstructed_image],
        max_rows=1,
        max_cols=1,
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        font_size=FONTSIZE_SMALL,
    )

    # Add axes title ({feat_1}^*, {feat_2}^*, {feat_3}^*) labeling the
    # fixed point, using the same stable fixed point marker color.
    ax = fig_fixed_point_reconstructions.axes[0]
    ax.set_title(
        f"({Unicode.THETA}$^*$, r$^*$, {Unicode.RHO}$^*$)",
        color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
        fontsize=FONTSIZE_XSMALL,
        fontweight="bold",
    )

    # add scalebars to each panel, only label the top left one to avoid
    # redundancy
    for i, ax in enumerate(fig_fixed_point_reconstructions.axes):
        add_scalebar(
            ax,
            scale_bar_um=20,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            bar_thickness=5,
            padding=5,
            include_label=True if i == 0 else False,
            label_fontsize=FONTSIZE_XSMALL,
        )

    dataset_name = fixed_point_df[Column.DATASET].unique().item()
    filename = f"{dataset_name}_fixed_point_reconstructions"
    save_plot_to_path(
        fig_fixed_point_reconstructions,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        bbox_inches="tight",
    )

    return output_path / f"{filename}.svg"


@figure_panel("Make panel of 3D vector field plot with stable fixed point overlay.")
def make_3d_vector_field_plot_panel(
    figure_size: tuple[float, float],
    output_path: Path,
    dataset_name: str,
    include_colorbar: bool = True,
    include_legend: bool = True,
) -> Path:
    """
    Render the 3D (theta, r, rho) drift vector field for a given dataset, with
    the stable fixed point overlaid as a scatter marker.

    Parameters
    ----------
    figure_size
        Size of the figure to create.
    output_path
        Directory in which to save the figure panel.
    dataset_name
        Name of the dataset to visualize.
    include_colorbar
        Whether to include a colorbar indicating the magnitude of the drift
        vectors.
    include_legend
        Whether to include a legend indicating the stable fixed point marker.

    Returns
    -------
    :
        Path to the saved figure file.

    """
    drift_dataframe = load_drift_dataframe_for_dataset(dataset_name)
    feature_dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    feature_dataframe = load_dataframe(feature_dataframe_manifest.locations[dataset_name])

    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    col_labels = [(COLUMN_METADATA[col].label or str(col)) for col in DYNAMICS_COLUMN_NAMES]
    fixed_point_label = f"({col_labels[0]}$^*$, {col_labels[1]}$^*$, {col_labels[2]}$^*$)"

    theta_lims = VECTOR_FIELD_THETA_RANGE
    r_lims = (0, 1.75)
    rho_lims = (-1.5, 1.5)

    # Load, clip, and downsample drift vector field
    drift, meshgrid = process_3d_vector_field_for_visualization(
        drift_dataframe,
        feature_dataframe,
        column_names=column_names,
        xlim=theta_lims,
        ylim=r_lims,
        zlim=rho_lims,
        mask_threshold=0.025,
    )

    fig, ax = plot_drift_3d(
        drift=drift,
        meshgrid=meshgrid,
        figsize=figure_size,
        include_colorbar=include_colorbar,
        include_legend=include_legend,
        fixed_point_legend_label=fixed_point_label,
        xlim=theta_lims,
        ylim=r_lims,
        zlim=rho_lims,
        xticks=[0, np.pi / 2],
        xtick_labels=[f"0={Unicode.PI}", f"{Unicode.PI}/2"],
        yticks=[0.25, 0.75, 1.25],
        zticks=[-1.0, 0, 1.0],
        xlabel=col_labels[0],
        ylabel=col_labels[1],
        zlabel=col_labels[2],
        xlabel_kwargs={"labelpad": -8},
        ylabel_kwargs={"labelpad": -5},
        zlabel_kwargs={"labelpad": -8},
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
        s=15,
        zorder=5,
    )

    # save as .svg file
    filename = f"3d_vector_field_{dataset_name}"
    save_plot_to_path(
        fig,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        bbox_inches="tight",
    )

    return output_path / f"{filename}.svg"


@figure_panel("Make panel of histogram of first passage time correlation values across datasets.")
def make_first_passage_time_correlation_hist(
    figure_size: tuple[float, float], output_path: Path, dataset_names: list[str]
) -> Path:
    fpt_manifest = load_dataframe_manifest(FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME)
    line_fit_df, _ = get_line_fit_and_filtered_df(fpt_manifest, dataset_names)

    pearson_correlations = []
    for _, df_dataset in line_fit_df.groupby(Column.DATASET):
        pearson_r = df_dataset[Column.VectorField.PEARSON_R].iloc[0]
        pearson_correlations.append(pearson_r)

    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
    ax.hist(pearson_correlations, bins=list(np.linspace(-1, 1, 21)), edgecolor="k")
    column_label = COLUMN_METADATA[Column.VectorField.PEARSON_R].label or str(
        Column.VectorField.PEARSON_R
    )
    ax.set_xlabel(column_label)
    ax.set_ylabel("Count")
    # make sure y ticks are integers since this is a count histogram
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    filename = "fpt_hist"
    save_plot_to_path(
        fig,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
    )
    return output_path / f"{filename}.svg"
