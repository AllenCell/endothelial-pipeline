"""Methods for visualizing the outputs of the DiffAE feature analysis workflows."""

from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D

from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_LEVELS,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.plot_defaults import SHEAR_COLOR_DICT


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
    x_values: np.ndarray,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[float, float] = (7, 4),
    axes_limits: list[tuple[float, float]] | None = None,
    axes_labels: list[str] | None = None,
    add_flow_arrows: bool = True,
    flow_arrow_downsample: int = 4,
    flow_arrow_kwargs: dict | None = None,
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
    fig, ax = fig_ax or plt.subplots(figsize=figsize, gridspec_kw=gridspec_kwargs)
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
        ax.quiver(
            x_values[::flow_arrow_downsample],
            y_values[::flow_arrow_downsample],
            drift[::flow_arrow_downsample],
            drift_y[::flow_arrow_downsample],
            **(flow_arrow_kwargs or {}),
        )

    if axes_limits is not None:
        ax.set_xlim(axes_limits[0])
        ax.set_ylim(axes_limits[1])
    if axes_labels is not None:
        ax.set_xlabel(axes_labels[0], **(xlabel_kwargs or {}))
        ax.set_ylabel(axes_labels[1], **(ylabel_kwargs or {}))

    return fig, ax


def plot_fixed_points_by_shear(
    fpt_dict_list: list, shear_range: np.ndarray, pcs: list, plt_lims: list
) -> tuple[list[plt.Figure], list[plt.Axes]]:
    """Plot individual components of fixed points as a function of shear stress.

    **Input dictionary format**:

    Each dictionary in `fpt_dict_list` should have the following keys:
        - "shear": float, the shear stress value corresponding to the fixed
          points in this dictionary
        - "fixed_points": list of np.ndarray, each array is a fixed point in the
          state space
        - "fixed_point_stability": list of str, each string is the stability
          type of the corresponding fixed point, e.g., "stable", "unstable",
          "saddle", or "indeterminate"

    Parameters
    ----------
    fpt_dict_list
        List of dictionaries, each containing fixed points, the corresponding
        types, and the shear stress value.
    shear_range
        Shear stress values corresponding to each dictionary in `fpt_dict_list`.
    pcs
        List of principal components used to fit the dynamical systems model.
    plt_lims
        List of tuples containing the axes y-limits for each plot.

    Returns
    -------
    :
        Tuple containing:
            - List of matplotlib Figure objects for each component plot.
            - List of corresponding Axes objects for each component plot.

    """
    if len(fpt_dict_list) != len(shear_range):
        raise ValueError(
            f"Length of fpt_dict_list ({len(fpt_dict_list)}) does not match length of shear_range ({len(shear_range)})."
        )

    # plot fixed points by shear stress
    # initialize figure and axes
    figs = []
    axs = []

    # loop over components
    ndim = len(pcs)
    for j in range(ndim):
        # initialize figure and axes for the j-th component
        fig, ax = plt.subplots(figsize=(7, 6))

        # loop over shear stress values
        for i, u in enumerate(shear_range):
            # get fixed points and types for the i-th shear stress value
            fpt_dict = fpt_dict_list[i]

            # check that the dict corresponds to the correct shear stress value
            if u != fpt_dict["shear"]:
                raise ValueError(
                    f"Shear stress value ({u}) does not match the value in the dictionary ({fpt_dict['shear']})."
                )

            # get fixed points and types
            fpts = fpt_dict["fixed_points"]
            fpt_stabilities = fpt_dict["fixed_point_stability"]

            # check that we have a type for each fixed point
            if len(fpts) != len(fpt_stabilities):
                raise ValueError(
                    f"Number of fixed points ({len(fpts)}) does not match number of "
                    f"stability types ({len(fpt_stabilities)}) for shear stress {u}."
                )

            # plot component j of each fixed point (if any)
            if len(fpts) > 0:
                # color code by type (stability)
                for ii, fpt in enumerate(fpts):
                    # default to black if type not in dict
                    color = SHEAR_COLOR_DICT.get(fpt_stabilities[ii], "k")
                    # plot
                    ax.plot(u, fpt[j], "o", color=color)
                    ax.set_xlabel("Shear stress (dyn/cm$^2$)")
                    ax.set_ylabel(f"PC{pcs[j] + 1}$^*$")
        # set fig title and limits, and append to lists
        ax.set_title("Fixed points by shear stress")
        ax.set_ylim(plt_lims[j])
        figs.append(fig)
        axs.append(ax)

    return figs, axs


def plot_histogram_2d(ax: plt.Axes, p_hist: np.ndarray, bins: list, cmap: str) -> plt.Axes:
    """Plot 2D histogram with specified colormap.

    Parameters
    ----------
    ax
        Axes to plot on.
    p_hist
        2D histogram data, e.g., obtained by np.histogram2d.
    bins
        List of bin edges used to compute the histogram for each dimension.
    cmap
        Colormap to use for the plot.

    Returns
    -------
    :
        The input Axes object with the histogram plotted on it.

    """
    # plot histogram, setting origin to lower left and
    # setting the aspect ratio to be square
    ax.imshow(
        p_hist.T,
        interpolation="nearest",
        origin="lower",
        extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]),
        cmap=cmap,
        aspect=(bins[0][-1] - bins[0][0]) / (bins[1][-1] - bins[1][0]),
    )

    # label axes
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    return ax


def kl_divergence(p: np.ndarray, q: np.ndarray, dx: list, tol: float = 1e-8) -> float:
    """Approximate Kullback-Leibler divergence between two (possibly multivariate) distributions.

    This method uses the formula `D_KL(p||q) = int p(x) log(p(x)/q(x)) dx`, where
    the integral is approximated by numerical integration (trapezoidal rule)
    over the grid defined by the bin edges corresponding to p and q.

    Parameters
    ----------
    p
        First probability distribution.
    q
        Second probability distribution.
    dx
        List of bin widths used to obtain the distributions for each dimension.
    tol
        Small value to avoid division by zero, by default 1e-8.

    Returns
    -------
    :
        The KL divergence D_KL(p||q) approximated by numerical integration.

    """
    ndim = len(dx)

    # set small values to tol
    p_ = p.copy()
    p_[p_ < tol] = tol
    q_ = q.copy()
    q_[q_ < tol] = tol

    kl_div = p_ * np.log(p_ / q_)  # initial KL divergence
    for i in range(ndim):
        kl_div = np.trapz(kl_div, dx=dx[i], axis=0)  # integrate over each dimension

    return kl_div


def compare_stationary_distributions(
    p_model: np.ndarray, p_hist: np.ndarray, bins: list
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """Compare predicted stationary distribution to histogram of data.

    This function creates a side-by-side plot of the histogram of the data at
    steady state (the "empirical PDF") and the numerical solution to the
    stationary Fokker-Planck equation for the fit SDE model (the "model PDF").

    The figure suptitle includes the Kullback-Leibler divergence between the two
    distributions, computed using numerical integration over the grid defined by
    the bin edges corresponding to p_hist and p_model. (See the `kl_divergence`
    function for details on the numerical approximation method used.)

    Parameters
    ----------
    p_model
        Predicted stationary distribution from the model, evaluated on the same
        grid as p_hist.
    p_hist
        Histogram of the data at steady state, evaluated on the same grid as
        p_model.
    bins
        List of bin edges used to compute the histogram for each dimension,
        which should be the same for p_hist and p_model.

    Returns
    -------
    :
        Tuple containing:
            - The matplotlib Figure object containing the side-by-side plots.
            - Array of the corresponding Axes objects for the empirical and model PDFs.

    """
    # check if 1D or 2D
    ndim = len(bins)
    if ndim == 2:  # call 2D histogram plot function
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0] = plot_histogram_2d(ax[0], p_hist, bins, cmap="inferno")  # plot empirical PDF
        ax[0].set_title("Empirical PDF")
        ax[1] = plot_histogram_2d(ax[1], p_model, bins, cmap="inferno")  # plot model PDF
        ax[1].set_title("Model PDF")

    elif ndim == 1:  # call 1D histogram plot function
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(bins[0][:-1], p_hist, "k", label="Empirical PDF")
        ax[0].set_title("Empirical PDF")
        ax[1].plot(bins[0][:-1], p_model, "k", label="Model PDF")
        ax[1].set_title("Model PDF")

    dx = [bins[i][1] - bins[i][0] for i in range(ndim)]  # bin widths
    kl_div = kl_divergence(p_hist, p_model, dx)

    fig.suptitle("$D_{KL}(p_{hist}||p_{model}) =$" + f"{kl_div:0.4f}", fontsize=16, y=1.05)

    return fig, ax


def plot_gen_potential_2d(
    potential: np.ndarray,
    xvec: np.ndarray,
    yvec: np.ndarray,
    cmap: str = "jet",
    surf: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2D generalized potential energy landscape with specified colormap.

    Parameters
    ----------
    potential
        2D array representing the generalized potential energy landscape,
        evaluated on the grid defined by xvec and yvec.
    xvec
        1D array of x-axis values corresponding to each point in the potential.
    yvec
        1D array of y-axis values corresponding to each point in the potential.
    cmap
        Colormap to use for the plot.
    surf
        Whether to plot as a surface (3D) or contour (2D). If True, plots a 3D
        surface; if False, plots a 2D contour.

    Returns
    -------
    :
        Tuple containing:
            - The matplotlib Figure object containing the plot.
            - The corresponding Axes object for the plot.

    """
    if surf:
        fig = plt.figure(figsize=plt.figaspect(1 / 3))
        ax: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")
        x_, y_ = np.meshgrid(xvec, yvec, indexing="ij")
        surf = ax.plot_surface(x_, y_, potential, cmap=cmap)
        ax.set_zlabel("$-\ln P$")
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            potential.T,
            interpolation="nearest",
            origin="lower",
            extent=(xvec[0], xvec[-1], yvec[0], yvec[-1]),
            cmap=cmap,
            aspect=(xvec[-1] - xvec[0]) / (yvec[-1] - yvec[0]),
        )
        fig.colorbar(im, label="$-\ln P$")
    return fig, ax


def plot_grad_flux_decomposition(
    potential: np.ndarray,
    xvec: np.ndarray,
    yvec: np.ndarray,
    grad: np.ndarray,
    flux: np.ndarray,
    cmap: str = "jet",
    normed: bool = False,
    downsample: int = 10,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot quiver plot of gradient and flux decomposition of a vector field.

    Parameters
    ----------
    potential
        2D array representing the generalized potential energy landscape,
        evaluated on the grid defined by xvec and yvec.
    xvec
        1D array of x-axis values corresponding to each point in the potential.
    yvec
        1D array of y-axis values corresponding to each point in the potential.
    grad
        3D array of shape (2, nx, ny) representing the gradient component of
        the vector field, evaluated on the same grid as potential.
    flux
        3D array of shape (2, nx, ny) representing the flux component of the
        vector field, evaluated on the same grid as potential.
    cmap
        Colormap to use for the potential energy landscape.
    normed
        If True, each vector is normalized to have unit length; if False, the
        vectors retain their original magnitude.
    downsample
        Factor by which to downsample the vectors for visualization.

    Returns
    -------
    :
        Tuple containing:
            - The matplotlib Figure object containing the plot.
            - The corresponding Axes object for the plot.

    """
    # contour plot of the potential energy landscape
    fig, ax = plot_gen_potential_2d(potential, xvec, yvec, cmap=cmap, surf=False)

    # quiver plot of gradient and flux decomposition
    # normalize vectors if specified
    if normed:
        grad = grad / (np.sqrt(grad[0] ** 2 + grad[1] ** 2))
        flux = flux / (np.sqrt(flux[0] ** 2 + flux[1] ** 2))

    # downsample vectors for visualization
    x_ = xvec[::downsample]
    y_ = yvec[::downsample]
    grad_ = grad[:, ::downsample, ::downsample]
    flux_ = flux[:, ::downsample, ::downsample]

    # plot quiver, color code by type (gradient or flux)
    ax.quiver(x_, y_, grad_[0].T, grad_[1].T, color="w", pivot="tail")
    ax.quiver(x_, y_, flux_[0].T, flux_[1].T, color="r", pivot="tail")

    return fig, ax
