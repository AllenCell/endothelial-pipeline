from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from endo_pipeline.library.visualize import viz_base


def _add_density_overlay(
    ax: plt.Axes,
    x_values: np.ndarray,
    density_values: np.ndarray,
    density_plot_color: str,
    density_plot_alpha: float,
    density_plot_edgecolor: str = "k",
    density_plot_linewidth: float = 0.5,
) -> plt.Axes:
    """
    Add density overlay to an existing Axes using seaborn kdeplot.

    Parameters
    ----------
    ax
        The Axes to add the density overlay to.
    x_values
        The values at which the density is evaluated.
    density
        The values of the density function.
    density_plot_color
        Color of the density plot overlay.
    density_plot_alpha
        Alpha (transparency) of the density plot overlay.
    density_plot_edgecolor
        Edge color of the density plot overlay.
    density_plot_linewidth
        Line width of the density plot overlay.
    """
    ax2 = ax.twinx()
    ax2.fill_between(
        x_values,
        density_values,
        color=density_plot_color,
        alpha=density_plot_alpha,
    )
    ax2.plot(
        x_values, density_values, color=density_plot_edgecolor, linewidth=density_plot_linewidth
    )
    ax2.set_ylabel("density")
    return ax


def plot_1d_drift(
    x_vals: np.ndarray,
    drift_vals: np.ndarray,
    variable_name: str,
    density: np.ndarray | None,
    drift_line_color: str = "k",
    drift_line_style: str = "-",
    zero_line_color: str = "r",
    zero_line_style: str = "--",
    zero_line_alpha: float = 0.5,
    density_plot_fillcolor: str = "gray",
    density_plot_alpha: float = 0.1,
    density_plot_edgecolor: str = "k",
    density_plot_linewidth: float = 0.5,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 1D drift coefficient function with optional overlay of the data density.

    E.g., for a 1D stochastic differential equation:
        dx = f(x)dt + g(x)dW_t,
    plot f(x) vs x.

    Parameters
    ----------
    x_vals
        Feature values where the drift is evaluated.
    drift_vals
        Corresponding drift values.
    variable_name
        Name of the variable being plotted (for labeling purposes).
    density
        Optional density function to overlay.
    drift_line_color
        Color of the line representing the drift function.
    drift_line_style
        Style of the line representing the drift function.
    zero_line_color
        Color of the zero reference line (y=0).
    zero_line_style
        Style of the zero reference line (y=0).
    zero_line_alpha
        Alpha (transparency) of the zero reference line (y=0).
    density_plot_fillcolor
        Fill color of the density plot overlay.
    density_plot_alpha
        Alpha (transparency) of the density plot overlay.
    density_plot_edgecolor
        Edge color of the density plot overlay.
    density_plot_linewidth
        Line width of the density plot overlay.
    """
    fig, ax = plt.subplots()
    # plot drift
    ax.plot(x_vals, drift_vals, f"{drift_line_color}{drift_line_style}")
    # draw zero line
    ax.plot(
        x_vals,
        np.zeros_like(x_vals),
        f"{zero_line_color}{zero_line_style}",
        alpha=zero_line_alpha,
        label="$y=0$",
    )
    ax.set_xlabel(f"${variable_name}$")
    ax.set_ylabel(f"drift in ${variable_name}$")

    # if data is provided, overlay density estimation using seaborn kdeplot
    if density is not None:
        ax = _add_density_overlay(
            ax,
            x_vals,
            density,
            density_plot_fillcolor,
            density_plot_alpha,
            density_plot_edgecolor,
            density_plot_linewidth,
        )

    return fig, ax


def plot_1d_diffusion(
    x_vals: np.ndarray,
    diffusion_vals: np.ndarray,
    variable_name: str,
    density: np.ndarray | None,
    diffusion_line_color: str = "k",
    diffusion_line_style: str = "-",
    mean_line_color: str = "b",
    mean_line_style: str = "--",
    mean_line_alpha: float = 0.5,
    density_plot_fillcolor: str = "gray",
    density_plot_alpha: float = 0.1,
    density_plot_edgecolor: str = "k",
    density_plot_linewidth: float = 0.5,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 1D diffusion coefficient function with optional overlay of the data density.

    E.g., for a 1D stochastic differential equation:
        dx = f(x)dt + g(x)dW_t,
    plot f(x) vs x.

    Parameters
    ----------
    x_vals
        Feature values where the drift is evaluated.
    diffusion_vals
        Corresponding drift values.
    variable_name
        Name of the variable being plotted (for labeling purposes).
    density
        Optional density function to overlay.
    diffusion_line_color
        Color of the line representing the drift function.
    diffusion_line_style
        Style of the line representing the drift function.
    mean_line_color
        Color of the line representing the mean diffusion value.
    mean_line_style
        Style of the line representing the mean diffusion value.
    mean_line_alpha
        Alpha (transparency) of the mean diffusion value line.
    density_plot_fillcolor
        Fill color of the density plot overlay.
    density_plot_alpha
        Alpha (transparency) of the density plot overlay.
    density_plot_edgecolor
        Edge color of the density plot overlay.
    density_plot_linewidth
        Line width of the density plot overlay.
    """
    fig, ax = plt.subplots()
    # plot drift
    ax.plot(x_vals, diffusion_vals, f"{diffusion_line_color}{diffusion_line_style}")

    # draw line = mean value of diffusion coefficient
    # where mean is weighted by the density of data points if provided
    if density is not None:
        diffusion_vals_mean = np.trapz(diffusion_vals * density, x=x_vals)
    else:
        diffusion_vals_mean = np.mean(diffusion_vals)

    ax.plot(
        x_vals,
        diffusion_vals_mean * np.ones_like(x_vals),
        f"{mean_line_color}{mean_line_style}",
        alpha=mean_line_alpha,
        label=f"$\\langle D({variable_name}) \\rangle$",
    )

    ax.set_xlabel(f"${variable_name}$")
    ax.set_ylabel(f"MSD in ${variable_name}$")

    # if data is provided, overlay density estimation using seaborn kdeplot
    if density is not None:
        ax = _add_density_overlay(
            ax,
            x_vals,
            density,
            density_plot_fillcolor,
            density_plot_alpha,
            density_plot_edgecolor,
            density_plot_linewidth,
        )

    # make sure y-limits start at 0
    ax.set_ylim((0.0, ax.get_ylim()[1]))
    return fig, ax


def plot_fixed_points_by_shear(
    fpt_dict_list: list, shear_range: np.ndarray, pcs: list, plt_lims: list
) -> tuple[list[plt.Figure], list[plt.Axes]]:
    """
    Plot individual components of fixed points (one for each dimension of the
    state space used to fit the dynamical systems model) of the system by shear stress.

    Input:
    - fpt_dict_list: list of dictionaries, each containing fixed points,
        the corresponding types, and the shear stress value
    - shear_range: np.ndarray, shear stress values corresponding
        to each dictionary in fpt_dict_list
    - PCs: list, list of principal components used to fit the dynamical systems model
    - plt_lims: list, list of tuples containing the limits for each plot

    Output:
    - figs: list of plt.Figure
    - axs: list of plt.Axes
    The length of figs and axs is equal to the number of principal components
    (i.e., the dimension of the state space). Figure i in figs corresponds
    to the plot of the i-th component of the identified fixed points.
    """
    assert len(fpt_dict_list) == len(shear_range)

    # plot fixed points by shear stress
    # initialize figure and axes
    figs = []
    axs = []

    # loop over components
    ndim = len(pcs)
    for j in range(ndim):
        # initialize figure and axes for the j-th component
        fig, ax = viz_base.init_plot()

        # loop over shear stress values
        for i, u in enumerate(shear_range):
            # get fixed points and types for the i-th shear stress value
            fpt_dict = fpt_dict_list[i]

            # check that the dict corresponds to the correct shear stress value
            assert u == fpt_dict["shear"]

            # get fixed points and types
            fpts = fpt_dict["fixed_points"]
            fpt_stabilities = fpt_dict["fixed_point_stability"]

            # check that we have a type for each fixed point
            assert len(fpts) == len(fpt_stabilities)

            # plot component j of each fixed point (if any)
            if len(fpts) > 0:
                # color code by type (stability)
                for ii, fpt in enumerate(fpts):
                    if fpt_stabilities[ii] == "stable":
                        color = "b"
                    elif fpt_stabilities[ii] == "unstable":
                        color = "r"
                    elif fpt_stabilities[ii] == "saddle":
                        color = "tab:purple"
                    else:  # can be indeterminate stability
                        color = "darkgoldenrod"
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
    """
    Plot 2D histogram with specified colormap.

    Input:
    - ax: plt.Axes, the axes to plot on
    - p_hist: np.ndarray, histogram data (e.g., obtained by np.histogram2d)
    - bins: list, list of bin edges used to compute the histogram for each dimension
    - cmap: str, colormap to use for the plot

    Output:
    - ax: plt.Axes
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
    """Approximate Kullback-Leibler divergence for arbitrary dimensionality."""
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
    """
    Side-by-side plots of the histogram of the data at steady state
    ("empirical PDF") and the numerical solution to the stationary
    Fokker-Planck equation for the fit SDE model ("model PDF").
    The figure suptitle includes K-L divergence between the two distributions.

    Input:
    - p_model: np.ndarray, model PDF (obtained from the numerical solution
        to the stationary Fokker-Planck equation)
    - p_hist: np.ndarray, empirical PDF (obtained from the data at
        steady state, e.g., by histogramming)
        - "steady state" here refers to the assumption that the
            data are stationary in some sense
    - bins: list, list of bin edges used to compute the p_hist for each dimension
        - should be the same as the bins used to compute p_model

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    # check if 1D or 2D
    ndim = len(bins)
    if ndim == 2:  # call 2D histogram plot function
        fig, ax = viz_base.init_subplots(figsize=(12, 4))
        ax[0] = plot_histogram_2d(ax[0], p_hist, bins, cmap="inferno")  # plot empirical PDF
        ax[0].set_title("Empirical PDF")
        ax[1] = plot_histogram_2d(ax[1], p_model, bins, cmap="inferno")  # plot model PDF
        ax[1].set_title("Model PDF")

    elif ndim == 1:  # call 1D histogram plot function
        fig, ax = viz_base.init_subplots(figsize=(12, 4))
        ax[0].plot(bins[0][:-1], p_hist, "k", label="Empirical PDF")
        ax[0].set_title("Empirical PDF")
        ax[1].plot(bins[0][:-1], p_model, "k", label="Model PDF")
        ax[1].set_title("Model PDF")

    dx = [bins[i][1] - bins[i][0] for i in range(ndim)]  # bin widths
    kl_div = kl_divergence(p_hist, p_model, dx)

    fig.suptitle("$D_{KL}(p_{hist}||p_{model}) =$" + f"{kl_div:0.4f}", fontsize=16, y=1.05)

    return fig, ax


def plot_entropy_production_rate(
    epr: np.ndarray, shear_range: np.ndarray
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot entropy production rate as a function of shear stress.

    Input:
    - epr: np.ndarray, entropy production rate values
    - shear_range: np.ndarray, shear stress values corresponding
        to each entropy production rate value

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    fig, ax = viz_base.init_plot()
    ax.plot(shear_range, epr, "-o", color="k")
    ax.set_xlabel("Shear stress (dyn/cm$^2$)")
    ax.set_ylabel("Entropy production rate")
    return fig, ax


def plot_gen_potential_2d(
    potential: np.ndarray,
    xvec: np.ndarray,
    yvec: np.ndarray,
    cmap: str = "jet",
    surf: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D generalized potential energy landscape with specified colormap.

    Input:
    - potential: np.ndarray, generalized potential energy landscape
    - xvec: np.ndarray, x-axis values corresponding to each point in U
    - yvec: np.ndarray, y-axis values corresponding to each point in U
    - cmap: str, colormap to use for the plot
    - surf: bool (default=False), whether to plot the surface as a 3D plot

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    if surf:
        fig = plt.figure(figsize=plt.figaspect(1 / 3))
        ax: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")
        x_, y_ = np.meshgrid(xvec, yvec, indexing="ij")
        surf = ax.plot_surface(x_, y_, potential, cmap=cmap)
        ax.set_zlabel("$-\ln P$")
        plt.tight_layout()
    else:
        fig, ax = viz_base.init_plot()
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
    """
    Plot quiver plot of gradient and flux decomposition of drift vector field
    over a contour plot of the 2D generalized potential energy landscape.

    Input:
    - potential: np.ndarray, generalized potential energy landscape
    - xvec: np.ndarray, x-axis values corresponding to each point in U
    - yvec: np.ndarray, y-axis values corresponding to each point in U
    - grad: np.ndarray, gradient part of the vector field
    - flux: np.ndarray, flux remainder part of the vector field
    - cmap: str (default='jet'), colormap to use for the plot
    - normed: bool (default=False), whether to normalize the gradient and
        flux vectors in the quiver plot
    - downsample: int (default=10), downsample factor for the quiver plot

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
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
