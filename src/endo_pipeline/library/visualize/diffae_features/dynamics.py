"""Methods for visualizing the outputs of the DiffAE feature analysis workflows."""

from collections.abc import Sequence
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from endo_pipeline.library.visualize.figure_utils import set_axes_properties
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_LEVELS,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, StabilityLegendHandle


def plot_drift_contours(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    fig_ax: tuple[plt.Figure, Sequence[plt.Axes]] | None = None,
    figsize: tuple[float, float] = (7, 12),
    n_rows: int = 2,
    n_cols: int = 1,
    axes_limits: list[tuple[float, float]] | None = None,
    axes_aspect: Literal["auto", "equal"] | float | None = "equal",
    axes_titles: tuple[str, str] | None = None,
    colormap: str = DRIFT_CONTOUR_COLORMAP,
    vmin: float | None = DRIFT_CONTOUR_VMIN,
    vmax: float | None = DRIFT_CONTOUR_VMAX,
    num_levels: int = DRIFT_CONTOUR_LEVELS,
    include_colorbar: bool = True,
    cbar_num_ticks: int = DRIFT_CONTOUR_CBAR_NUM_TICKS,
    cbar_tick_round: int = DRIFT_CONTOUR_CBAR_ROUND,
    include_nullclines: bool = True,
    nullcline_styles: tuple = ("dashed", "dashdot"),
    nullcline_colors: tuple = ("k", "k"),
    nullcline_linewidth: float = 1.5,
    nullcline_opacity: float = 0.7,
    gridspec_kwargs: dict | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
    axes_title_kwargs: dict | None = None,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Make and save contour plot of each component of the drift vector field over
    the 2D state space.

    The contour lines are colored according to the value of the drift component,
    using a diverging colormap centered at zero to visualize the direction and
    magnitude of the drift.

    Parameters
    ----------
    meshgrid
        Meshgrid on which the drift is evaluated, typically obtained from
        np.meshgrid(..., indexing="ij").
    drift
        Drift vector field evaluated on the meshgrid, with shape (nx, ny, ndim).
    variable_labels
        Labels for axes corresponding to the state space variables, e.g.,
        ["$x_1$", "$x_2$"].
    fig_ax
        Optional tuple of figure and axes objects to plot on. If None, a new
        figure and axes will be created; if provided, the contour plots will be
        made on the provided axes.
    figsize
        Size of the figure, specified as a tuple (width, height).
    axes_limits
        Optional limits for the axes, specified as a list of tuples.
    axes_aspect
        Aspect ratio for the axes, e.g., "equal" to make x and y have equal scaling.
    axes_titles
        Optional tuple of titles for each subplot.
    colormap
        Colormap to use for the contour plots.
    vmin
        Optional, minimum colorbar value for the contour plots.
    vmax
        Optional, maximum colorbar value for the contour plots.
    num_levels
        Number of contour levels to use in the plot.
    include_colorbar
        Whether to include a colorbar for each contour plot.
    cbar_num_ticks
        Number of ticks to use in the colorbar for each contour plot.
    cbar_tick_round
        Number of decimal places to round colorbar ticks to in the contour plots.
    nullcline_styles
        Tuple of line styles for the nullclines of each variable.
    nullcline_colors
        Tuple of colors for the nullclines of each variable.
    nullcline_linewidth
        Line width for the nullcline lines.
    nullcline_opacity
        Opacity for the nullcline lines (between 0 and 1).
    gridspec_kwargs
        Optional dictionary of keyword arguments to pass to plt.subplots for
        creating the figure and axes, e.g., to specify a GridSpec layout.
    xlabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_xlabel for
        customizing the x-axis label, e.g., to specify a font size or label padding.
    ylabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_ylabel for
        customizing the y-axis label, e.g., to specify a font size or label padding.
    axes_title_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_title for
        customizing the subplot titles, e.g., to specify a font size.

    """
    fig, ax = fig_ax or plt.subplots(
        n_rows, n_cols, figsize=figsize, layout="constrained", gridspec_kw=gridspec_kwargs
    )
    ax = cast(
        Sequence[plt.Axes], ax
    )  # for type checking, since ax is either a single Axes or a sequence of Axes

    for var_index, var_name in enumerate(variable_labels):
        vmin_ = vmin or np.nanmin(drift[..., var_index])
        vmax_ = vmax or np.nanmax(drift[..., var_index])
        contour_levels = np.linspace(vmin_, vmax_, num_levels)
        # center colormap at zero to visualize sign and magnitude of drift
        colormap_norm = TwoSlopeNorm(vmin=vmin_, vmax=vmax_, vcenter=0)

        contour = ax[var_index].contourf(
            meshgrid[0],
            meshgrid[1],
            drift[..., var_index],
            levels=contour_levels,
            cmap=colormap,
            norm=colormap_norm,
            extend="both",
        )
        if include_nullclines:
            # add dashed line for nullcline
            ax[var_index].contour(
                meshgrid[0],
                meshgrid[1],
                drift[..., var_index],
                levels=[0],
                colors=nullcline_colors[var_index],
                linestyles=[nullcline_styles[var_index]],
                linewidths=nullcline_linewidth,
                alpha=nullcline_opacity,
            )
        if include_colorbar:
            colorbar_ticks = np.linspace(vmin_, vmax_, cbar_num_ticks)
            colorbar_ticks = np.round(colorbar_ticks, cbar_tick_round)
            fig.colorbar(contour, ax=ax[var_index], label=f"d{var_name}/dt", ticks=colorbar_ticks)

        # set axis properties, only including label for edge plot of shared axes
        # (e.g., only xlabel for left column and only ylabel for bottom row if
        # multiple rows/columns of subplots)
        xlabel: str | None
        ylabel: str | None
        if n_rows > n_cols:
            # if more rows than columns, only set xlabel for bottom row
            xlabel = variable_labels[0] if var_index == n_rows - 1 else None
            ylabel = variable_labels[1]
        elif n_cols >= n_rows:
            # if more columns than rows, only set ylabel for left column
            xlabel = variable_labels[0]
            ylabel = variable_labels[1] if var_index == 0 else None
        set_axes_properties(
            ax[var_index],
            xlim=axes_limits[0] if axes_limits else None,
            ylim=axes_limits[1] if axes_limits else None,
            xlabel=xlabel,
            ylabel=ylabel,
            title=axes_titles[var_index] if axes_titles else None,
            aspect=axes_aspect,
            xlabel_kwargs=xlabel_kwargs,
            ylabel_kwargs=ylabel_kwargs,
            title_kwargs=axes_title_kwargs,
        )

    return fig, ax


def plot_drift_quiver(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[float, float] = (4, 4),
    axes_limits: list[tuple[float, float]] | None = None,
    quiver_scale: float = 4,
    quiver_color: str = "dimgrey",
    quiver_downsample: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    include_nullclines: bool = True,
    nullcline_styles: tuple = ("dashed", "dashdot"),
    nullcline_colors: tuple = ("k", "k"),
    nullcline_linewidth: float = 1.5,
    nullcline_opacity: float = 0.9,
    gridspec_kwargs: dict | None = None,
    legend_kwargs: dict | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
    plot_legend: bool = True,
):
    """
    Make and save quiver plot of the drift vector field over the 2D state space.

    Parameters
    ----------
    meshgrid
        Meshgrid on which the drift is evaluated, typically obtained from
        np.meshgrid(..., indexing="ij").
    drift
        Drift vector field evaluated on the meshgrid, with shape (nx, ny, ndim).
    variable_labels
        Labels for axes corresponding to the state space variables, e.g.,
        ["$x_1$", "$x_2$"].
    fig_ax
        Optional tuple of (Figure, Axes) to plot on. If None, a new figure and
        axes will be created; if provided, the quiver plot will be made on the
        provided axes.
    figsize
        Size of the figure, specified as a tuple (width, height).
    axes_limits
        Optional limits for the axes, specified as a list of tuples.
    quiver_scale
        Scale factor for the quiver plot (smaller values make arrows longer).
    quiver_color
        Color for the quiver arrows.
    quiver_downsample
        Factor by which to downsample the quiver arrows for visualization.
    include_nullclines
        Whether to include nullclines (where drift components are zero).
    nullcline_styles
        Tuple of line styles for the nullclines of each variable.
    nullcline_colors
        Tuple of colors for the nullclines of each variable.
    nullcline_linewidth
        Line width for the nullcline lines.
    nullcline_opacity
        Opacity for the nullcline lines (between 0 and 1).
    gridspec_kwargs
        Optional dictionary of keyword arguments to pass to plt.subplots for
        creating the figure and axes, e.g., to specify a GridSpec layout.
    legend_kwargs
        Optional dictionary of keyword arguments to pass to ax.legend for
        customizing the legend, e.g., to specify a title or font size.
    xlabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_xlabel for
        customizing the x-axis label, e.g., to specify a font size or label padding.
    ylabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_ylabel for
        customizing the y-axis label, e.g., to specify a font size or label padding.
    plot_legend
        Whether to plot the legend for the nullclines.
    """

    # if vmin and vmax are provided, rescale components of the drift to be
    # between vmin and vmax for visualization purposes (e.g., to make arrows
    # more visible if drift magnitudes are very small or very large)
    if vmin is not None and vmax is not None:
        for component in range(drift.shape[-1]):
            drift[..., component] = np.clip(drift[..., component], vmin, vmax)

    fig, ax = fig_ax or plt.subplots(figsize=figsize, gridspec_kw=gridspec_kwargs)
    ax.quiver(
        meshgrid[0][::quiver_downsample, ::quiver_downsample],
        meshgrid[1][::quiver_downsample, ::quiver_downsample],
        drift[::quiver_downsample, ::quiver_downsample, 0],
        drift[::quiver_downsample, ::quiver_downsample, 1],
        color=quiver_color,
        pivot="tail",
        scale=quiver_scale,
    )
    if include_nullclines:
        for var_index, var_name in enumerate(variable_labels):
            # add dashed line for nullcline
            ax.contour(
                meshgrid[0],
                meshgrid[1],
                drift[..., var_index],
                levels=[0],
                colors=nullcline_colors[var_index],
                linestyles=[nullcline_styles[var_index]],
                linewidths=nullcline_linewidth,
                alpha=nullcline_opacity,
            )
            # add legend for nullclines
            ax.plot(
                [],
                [],
                color=nullcline_colors[var_index],
                linestyle=nullcline_styles[var_index],
                label=f"Nullcline d{var_name}/dt=0",
            )
        if plot_legend:
            ax.legend(**(legend_kwargs or {}))

    set_axes_properties(
        ax,
        xlim=axes_limits[0] if axes_limits else None,
        ylim=axes_limits[1] if axes_limits else None,
        xlabel=variable_labels[0],
        ylabel=variable_labels[1],
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )

    return fig, ax


def plot_drift_1d(
    drift: np.ndarray,
    x_values: np.ndarray,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[float, float] = (4, 4),
    axes_limits: list[tuple[float, float]] | None = None,
    axes_labels: list[str] | None = None,
    add_flow_arrows: bool = True,
    flow_arrow_downsample: int = 5,
    flow_arrow_kwargs: dict | None = {"color": "dimgrey"},
    gridspec_kwargs: dict | None = None,
    drift_line_kwargs: dict | None = None,
    zero_line_kwargs: dict | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 1D drift as a function of the state variable.

    Parameters
    ----------
    drift
        1D array of the drift component evaluated at the input `x_values`.
    x_values
        1D array of state variable values corresponding to the drift values.
    fig_ax
        Optional tuple of (Figure, Axes) to plot on. If None, a new figure and
        axes will be created; if provided, the plot will be made on the provided
        axes.
    figsize
        Size of the figure, specified as a tuple (width, height).
    axes_limits
        Optional limits for the axes, specified as a list of tuples.
    axes_labels
        Optional list of labels for the x and y axes, specified as a list of
        strings.
    add_flow_arrows
        If true, draw arrows along y = 0 to indicate the direction of flow,
        pointing right if drift is positive and left if drift is negative.
    flow_arrow_downsample
        Integer specifying the downsampling factor for the flow arrows. Arrows
        will be drawn at every nth center, where n is the downsampling factor.
    flow_arrow_kwargs
        Optional dictionary of keyword arguments to pass to ax.arrow for
        customizing the appearance of the flow arrows, e.g., to specify color or
        line width.
    gridspec_kwargs
        Optional dictionary of keyword arguments to pass to plt.subplots for
        creating the figure and axes, e.g., to specify a GridSpec layout.
    drift_line_kwargs
        Dictionary of keyword arguments to pass to ax.plot for customizing the
        line representing the drift, e.g., to specify color or line width.
    zero_line_kwargs
        Dictionary of keyword arguments to pass to ax.plot for customizing the
        line representing the zero drift level, e.g., to specify color, line
        style, line width, or opacity.
    xlabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_xlabel for
        customizing the x-axis label, e.g., to specify a font size or label
        padding.
    ylabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_ylabel for
        customizing the y-axis label, e.g., to specify a font size or label
        padding.

    Returns
    -------
    :
        Tuple of figure and axes objects containing the plot of the 1D drift as
        a function of the state variable.
    """
    fig, ax = fig_ax or plt.subplots(
        figsize=figsize, layout="constrained", gridspec_kw=gridspec_kwargs
    )
    ax.plot(x_values, drift, **(drift_line_kwargs or {}))
    ax.plot(x_values, np.zeros_like(x_values), **(zero_line_kwargs or {}))

    # add arrows to indicate flow direction, pointing right if drift is
    # positive and left if drift is negative (downsampled for vis)
    if add_flow_arrows:
        # make stand-in y values and drift in y direction for quiver plot: plot
        # arrows along y=0, with length and direction determined by drift values
        # in x (with "drift" in y = 0)
        y_values = np.zeros_like(x_values)
        drift_y = np.zeros_like(drift)

        # if scale is not specified in flow_arrow_kwargs, set it automatically
        # based on the maximum absolute value of the drift and the space between
        # arrows, to make arrow lengths visually informative without being too
        # small or too large
        if flow_arrow_kwargs is None or "scale" not in flow_arrow_kwargs:
            max_drift = np.max(np.abs(drift))
            downsample_spacing = np.mean(np.diff(x_values[::flow_arrow_downsample]))
            if max_drift > 0:
                flow_arrow_kwargs = flow_arrow_kwargs or {}
                flow_arrow_kwargs["scale"] = max_drift / downsample_spacing * 0.75
            else:
                flow_arrow_kwargs = flow_arrow_kwargs or {}
                flow_arrow_kwargs["scale"] = 1.0

        ax.quiver(
            x_values[::flow_arrow_downsample],
            y_values[::flow_arrow_downsample],
            drift[::flow_arrow_downsample],
            drift_y[::flow_arrow_downsample],
            **(flow_arrow_kwargs or {}),
        )

    set_axes_properties(
        ax,
        xlim=axes_limits[0] if axes_limits else None,
        ylim=axes_limits[1] if axes_limits else None,
        xlabel=axes_labels[0] if axes_labels else None,
        ylabel=axes_labels[1] if axes_labels else None,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )

    return fig, ax


def make_legend_handles_for_fixed_pts(
    fpt_stabilities: list[str],
    marker_size: int = 10,
    edge_color: str = "black",
) -> list[StabilityLegendHandle]:
    """Make a custom legend for the fixed point types, nullclines and trajectories.

    Purpose of this method is to create a legend that only includes the fixed
    point types that are present in the plot, since the number and type of fixed
    points can vary across parameter space. That is, we want to avoid having
    duplicate labels where we have multiple fixed points of the same type, but
    we also want to avoid having labels for types that are not present.

    Parameters
    ----------
    fpt_stabilities
        List of stability labels for the fixed points.
    marker_size
        Size of the markers for the legend handles.
    edge_color
        Color of the marker edges.

    Returns
    -------
    :
        List of StabilityLegendHandle objects representing the legend handles
        for the fixed point types.

    """
    my_handles = []
    # get legend handles for the fixed point types that are present in given
    # list of fixed point stabilities, in the order given by StabilityLabel enum
    for stability_type in StabilityLabel:
        if stability_type in fpt_stabilities:
            my_handles.append(
                StabilityLegendHandle(
                    stability_label=stability_type,
                    legend_label=f"{stability_type} fixed point",
                    marker=FIXED_POINT_PLOT_STYLE[stability_type].marker,
                    face_color=FIXED_POINT_PLOT_STYLE[stability_type].color,
                    edge_color=edge_color,
                    marker_size=marker_size,
                )
            )

    return my_handles
