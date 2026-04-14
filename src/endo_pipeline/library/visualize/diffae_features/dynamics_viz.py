"""Methods for visualizing the outputs of the DiffAE feature analysis workflows."""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D

from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
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
    fig_ax: tuple[plt.Figure, tuple[plt.Axes, plt.Axes]] | None = None,
    figsize: tuple[float, float] = (MAX_FIGURE_WIDTH, 2 * MAX_FIGURE_WIDTH),
    axes_limits: list[tuple[float, float]] | None = None,
    axes_aspect: Literal["auto", "equal"] | float | None = "equal",
    axes_titles: list[str] | None = None,
    colormap: str = DRIFT_CONTOUR_COLORMAP,
    vmin: float | None = DRIFT_CONTOUR_VMIN,
    vmax: float | None = DRIFT_CONTOUR_VMAX,
    num_levels: int = DRIFT_CONTOUR_LEVELS,
    include_colorbar: bool = True,
    cbar_num_ticks: int = DRIFT_CONTOUR_CBAR_NUM_TICKS,
    cbar_tick_round: int = DRIFT_CONTOUR_CBAR_ROUND,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
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
        Optional tuple of (Figure, (Axes, Axes)) to plot on. If None, a new
        figure and axes will be created; if provided, the contour plots will be
        made on the provided axes.
    figsize
        Size of the figure, specified as a tuple (width, height).
    axes_limits
        Optional limits for the axes, specified as a list of tuples.
    axes_aspect
        Aspect ratio for the axes, e.g., "equal" to make x and y have equal scaling.
    axes_titles
        Optional list of titles for each subplot.
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

    """
    fig, ax = fig_ax or plt.subplots(2, 1, figsize=figsize)

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
        # add dashed line for nullcline
        ax[var_index].contour(
            meshgrid[0],
            meshgrid[1],
            drift[..., var_index],
            levels=[0],
            colors="k",
            linestyles="dashed",
        )
        if include_colorbar:
            colorbar_ticks = np.linspace(vmin_, vmax_, cbar_num_ticks)
            colorbar_ticks = np.round(colorbar_ticks, cbar_tick_round)
            fig.colorbar(contour, ax=ax[var_index], label=f"d{var_name}/dt", ticks=colorbar_ticks)
        ax[var_index].set_xlabel(variable_labels[0])
        ax[var_index].set_ylabel(variable_labels[1])
        if axes_limits:
            ax[var_index].set_xlim(axes_limits[0])
            ax[var_index].set_ylim(axes_limits[1])
        if axes_titles:
            ax[var_index].set_title(axes_titles[var_index])
        if axes_aspect:
            ax[var_index].set_aspect(axes_aspect)
    return fig, ax


def plot_drift_quiver(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[float, float] = (MAX_FIGURE_HEIGHT / 2, MAX_FIGURE_HEIGHT / 3),
    axes_limits: list[tuple[float, float]] | None = None,
    quiver_scale: float = 10,
    quiver_color: str = "k",
    include_nullclines: bool = True,
    legend_loc: tuple[float, float] = (1.05, 0.80),
    nullcline_styles: tuple[str, str] = ("dashed", "dashdot"),
    nullcline_colors: tuple[str, str] = ("r", "b"),
    nullcline_linewidth: float = 1.5,
    nullcline_opacity: float = 0.7,
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

    """
    fig, ax = fig_ax or plt.subplots(figsize=figsize)
    ax.quiver(
        meshgrid[0],
        meshgrid[1],
        drift[..., 0],
        drift[..., 1],
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
                linestyles=nullcline_styles[var_index],
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
        ax.legend(title="Nullclines", loc=legend_loc)

    ax.set_xlabel(variable_labels[0])
    ax.set_ylabel(variable_labels[1])
    if axes_limits:
        ax.set_xlim(axes_limits[0])
        ax.set_ylim(axes_limits[1])

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
