import matplotlib.pyplot as plt
import numpy as np

from cellsmap.analyses.utils.viz import viz_base as vb


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
        fig, ax = vb.init_plot()

        # loop over shear stress values
        for i, u in enumerate(shear_range):
            # get fixed points and types for the i-th shear stress value
            fpt_dict = fpt_dict_list[i]

            # check that the dict corresponds to the correct shear stress value
            assert u == fpt_dict["shear"]

            # get fixed points and types
            fpts = fpt_dict["fixed_points"]
            fpt_types = fpt_dict["fixed_point_types"]

            # check that we have a type for each fixed point
            assert len(fpts) == len(fpt_types)

            # plot component j of each fixed point (if any)
            if len(fpts) > 0:
                # color code by type (stability)
                for ii, fpt in enumerate(fpts):
                    if fpt_types[ii] == "stable":
                        color = "b"
                    elif fpt_types[ii] == "unstable":
                        color = "r"
                    elif fpt_types[ii] == "saddle":
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


def plot_histogram_2d(
    ax: plt.Axes, p_hist: np.ndarray, bins: list, cmap: str
) -> plt.Axes:
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
        extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
        cmap=cmap,
        aspect=(bins[0][-1] - bins[0][0]) / (bins[1][-1] - bins[1][0]),
    )

    # label axes
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    return ax


def kl_divergence(p, q, dx, tol=1e-8):
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
    p_model: np.ndarray, p_hist: np.ndarray, bins
) -> tuple[plt.Figure, plt.Axes]:
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
        fig, ax = vb.init_subplots(figsize=(12, 4))
        ax[0] = plot_histogram_2d(
            ax[0], p_hist, bins, cmap="inferno"
        )  # plot empirical PDF
        ax[0].set_title("Empirical PDF")
        ax[1] = plot_histogram_2d(
            ax[1], p_model, bins, cmap="inferno"
        )  # plot model PDF
        ax[1].set_title("Model PDF")

    elif ndim == 1:  # call 1D histogram plot function
        fig, ax = vb.init_subplots(figsize=(12, 4))
        ax[0].plot(bins[0][:-1], p_hist, "k", label="Empirical PDF")
        ax[0].set_title("Empirical PDF")
        ax[1].plot(bins[0][:-1], p_model, "k", label="Model PDF")
        ax[1].set_title("Model PDF")

    dx = [bins[i][1] - bins[i][0] for i in range(ndim)]  # bin widths
    kl_div = kl_divergence(p_hist, p_model, dx)

    fig.suptitle(
        "$D_{KL}(p_{hist}||p_{model}) =$" + f"{kl_div:0.4f}", fontsize=16, y=1.05
    )

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
    fig, ax = vb.init_plot()
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
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        x_, y_ = np.meshgrid(xvec, yvec, indexing="ij")
        surf = ax.plot_surface(x_, y_, potential, cmap=cmap)
        ax.set_zlabel("$-\ln P$")
        plt.tight_layout()
    else:
        fig, ax = vb.init_plot()
        im = ax.imshow(
            potential.T,
            interpolation="nearest",
            origin="lower",
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            cmap=cmap,
            aspect=(xvec[-1] - xvec[0]) / (yvec[-1] - yvec[0]),
        )
        fig.colorbar(im, label="$-\ln P$")
    return fig, ax


def plot_grad_flux_decomposition(
    potential: np.ndarray,
    xvec: np.ndarray,
    yvec: np.ndarray,
    grad,
    flux,
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
