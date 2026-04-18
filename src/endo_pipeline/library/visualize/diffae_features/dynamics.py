"""Methods for visualizing the outputs of the DiffAE feature analysis workflows."""

from collections.abc import Sequence
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_LEVELS,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)


def plot_drift_contours(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    fig_ax: tuple[plt.Figure, Sequence[plt.Axes]] | None = None,
    figsize: tuple[float, float] = (7, 12),
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
    nullcline_colors: tuple = ("r", "b"),
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
    fig, ax = fig_ax or plt.subplots(2, 1, figsize=figsize, gridspec_kw=gridspec_kwargs)
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
        if var_index == 1:
            # add shared x-axis label only for the second subplot
            ax[var_index].set_xlabel(variable_labels[0], **(xlabel_kwargs or {}))
        ax[var_index].set_ylabel(variable_labels[1], **(ylabel_kwargs or {}))
        if axes_limits:
            ax[var_index].set_xlim(axes_limits[0])
            ax[var_index].set_ylim(axes_limits[1])
        if axes_titles:
            ax[var_index].set_title(axes_titles[var_index], **(axes_title_kwargs or {}))
        if axes_aspect:
            ax[var_index].set_aspect(axes_aspect)

    return fig, ax


def plot_contour_colorbar(
    figsize: tuple[float, float] = (2, 4),
    label: str | None = None,
    vmin: float = DRIFT_CONTOUR_VMIN,
    vmax: float = DRIFT_CONTOUR_VMAX,
    num_ticks: int = DRIFT_CONTOUR_CBAR_NUM_TICKS,
    tick_label_round: int = DRIFT_CONTOUR_CBAR_ROUND,
    center_zero: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    extend: Literal["neither", "both", "min", "max"] = "both",
    colormap: str = DRIFT_CONTOUR_COLORMAP,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot contour plot colorbar as a standalone figure.

    Parameters
    ----------
    figsize
        Size of the figure for the colorbar, specified as a tuple (width,
        height).
    label
        Label for the colorbar, e.g., "d$x_1$/dt".
    vmin
        Minimum value for the colorbar.
    vmax
        Maximum value for the colorbar.
    num_ticks
        Number of ticks to use in the colorbar.
    tick_label_round
        Number of decimal places to round colorbar tick labels to.
    center_zero
        Whether to center the colormap at zero.
    orientation
        Orientation of the colorbar, either "vertical" or "horizontal".
    colormap
        Colormap to use for the colorbar, which should match the colormap used
        in the contour plot.

    Returns
    -------
    :
        The input Axes object with the colorbar added to it.

    """
    fig, ax = plt.subplots(figsize=figsize)

    if label:
        ax.set_title(label)

    if center_zero:
        color_mappable = ScalarMappable(
            norm=TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0), cmap=colormap
        )
    else:
        color_mappable = ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=colormap)

    colorbar_ticks = np.linspace(vmin, vmax, num_ticks)
    colorbar_ticks = np.round(colorbar_ticks, tick_label_round)

    fig.colorbar(
        color_mappable,
        cax=ax,
        orientation=orientation,
        ticks=colorbar_ticks,
        extend=extend,
    )
    return fig, ax


def plot_drift_quiver(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[float, float] = (4, 4),
    axes_limits: list[tuple[float, float]] | None = None,
    quiver_scale: float = 10,
    quiver_color: str = "k",
    quiver_downsample: int = 1,
    include_nullclines: bool = True,
    nullcline_styles: tuple = ("dashed", "dashdot"),
    nullcline_colors: tuple = ("r", "b"),
    nullcline_linewidth: float = 1.5,
    nullcline_opacity: float = 0.7,
    gridspec_kwargs: dict | None = None,
    legend_kwargs: dict | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
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

    """
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
                label=f"d{var_name}/dt",
            )
        ax.legend(title="Nullclines", **(legend_kwargs or {}))

    ax.set_xlabel(variable_labels[0], **(xlabel_kwargs or {}))
    ax.set_ylabel(variable_labels[1], **(ylabel_kwargs or {}))
    if axes_limits:
        ax.set_xlim(axes_limits[0])
        ax.set_ylim(axes_limits[1])

    return fig, ax


def plot_drift_1d(
    drift: np.ndarray,
    centers: np.ndarray,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[float, float] = (7, 4),
    axes_limits: list[tuple[float, float]] | None = None,
    axes_labels: list[str] | None = None,
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
        1D array of the drift component evaluated at the corresponding centers.
    centers
        1D array of the centers of the bins corresponding to the drift values.
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
    fig, ax = fig_ax or plt.subplots(figsize=figsize, gridspec_kw=gridspec_kwargs)
    ax.plot(centers, drift, **(drift_line_kwargs or {}))
    ax.plot(centers, np.zeros_like(centers), **(zero_line_kwargs or {}))

    if axes_limits is not None:
        ax.set_xlim(axes_limits[0])
        ax.set_ylim(axes_limits[1])
    if axes_labels is not None:
        ax.set_xlabel(axes_labels[0], **(xlabel_kwargs or {}))
        ax.set_ylabel(axes_labels[1], **(ylabel_kwargs or {}))

    return fig, ax
