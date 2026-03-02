from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

from endo_pipeline.io import save_plot_to_path


def plot_and_save_drift_contours(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    bin_limits: list[tuple[float, float]],
    fig_title: str,
    fig_savedir: Path,
    filename_prefix: str,
) -> None:
    """
    Plot contour of each component of the drift vector field over the 2D state
    space.

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
    bin_limits
        Limits for the axes, specified as a list of tuples.
    fig_title
        Title for the figure.
    fig_savedir
        Directory to save the figure.
    filename_prefix
        Prefix for the filename when saving the figure, e.g., "dataset_1".
    """
    for var_index, var_name in enumerate(variable_labels):
        fig, ax = plt.subplots()
        contour = ax.contourf(
            meshgrid[0],
            meshgrid[1],
            drift[..., var_index],
            levels=50,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0),
        )
        # add dashed line for nullcline
        ax.contour(
            meshgrid[0],
            meshgrid[1],
            drift[..., var_index],
            levels=[0],
            colors="k",
            linestyles="dashed",
        )
        fig.colorbar(contour, ax=ax, label=f"d{var_name}/dt")
        ax.set_xlabel(variable_labels[0])
        ax.set_ylabel(variable_labels[1])
        ax.set_xlim(bin_limits[0])
        ax.set_ylim(bin_limits[1])
        fig.suptitle(
            f"{fig_title} \n d{var_name}/dt vs ({variable_labels[0]}, {variable_labels[1]})", y=1.05
        )
        var_name_for_file = var_name.replace("$", "").replace("\\", "")
        save_plot_to_path(fig, fig_savedir, f"{filename_prefix}_d{var_name_for_file}dt")


def plot_and_save_drift_quiver(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    bin_limits: list[tuple[float, float]],
    fig_title: str,
    fig_savedir: Path,
    filename_prefix: str,
    include_nullclines: bool = True,
    quiver_scale: float = 10,
    quiver_color: str = "k",
    nullcline_styles: tuple[str, str] = ("dashed", "dashdot"),
    nullcline_colors: tuple[str, str] = ("r", "b"),
    nullcline_linewidth: float = 1.5,
    nullcline_opacity: float = 0.7,
):
    """
    Plot quiver plot of the drift vector field over the 2D state space.

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
    bin_limits
        Limits for the axes, specified as a list of tuples.
    fig_title
        Title for the figure.
    fig_savedir
        Directory to save the figure.
    filename_prefix
        Prefix for the filename when saving the figure, e.g., "dataset_1".
    include_nullclines
        Whether to include nullclines (where drift components are zero).
    quiver_scale
        Scale factor for the quiver plot (smaller values make arrows longer).
    quiver_color
        Color for the quiver arrows.
    nullcline_styles
        Tuple of line styles for the nullclines of each variable.
    nullcline_colors
        Tuple of colors for the nullclines of each variable.
    nullcline_linewidth
        Line width for the nullcline lines.
    nullcline_opacity
        Opacity for the nullcline lines (between 0 and 1).
    """
    fig, ax = plt.subplots()
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
        ax.legend(title="Nullclines", loc=(1.025, 0.90))

    ax.set_xlabel(variable_labels[0])
    ax.set_ylabel(variable_labels[1])
    ax.set_xlim(bin_limits[0])
    ax.set_ylim(bin_limits[1])
    fig.suptitle(f"{fig_title} \n drift in ({variable_labels[0]}, {variable_labels[1]})", y=1.05)
    save_plot_to_path(fig, fig_savedir, f"{filename_prefix}_drift_quiver")


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
        fig, ax = plt.subplots(figsize=(7, 6))

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
    fig, ax = plt.subplots(figsize=(7, 6))
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


def plot_population_cov_vs_time(
    pop_cov_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
    ylim_dict: dict[str, tuple[float, float]] | None = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot population CoV vs time for all dataset / flow conditions on a shared figure.

    Each dataset-condition is drawn as a separate line and coloured by shear stress
    regime.  A single shared legend is placed below the subplots.

    Parameters
    ----------
    pop_cov_data
        Mapping from feature column name to a list of
        ``(time_values, cov_series, color, label)`` tuples — one per
        dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    ylim_dict
        Optional mapping from feature column name to ``(ymin, ymax)`` y-axis
        limits.  When ``None`` (default), matplotlib auto-scales the y-axis.
        Columns absent from the dict also fall back to auto-scaling.
    """
    column_names = list(pop_cov_data.keys())
    n_cols = len(column_names)

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

    for col, ax in zip(column_names, axs, strict=False):
        for time_values, cov_series, color, label in pop_cov_data[col]:
            ax.plot(time_values, cov_series, color=color, label=label, alpha=0.75, linewidth=1.2)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"{variable_labels_dict[col]} CoV")
        ax.set_title(variable_labels_dict[col])
        if ylim_dict is not None and col in ylim_dict:
            ax.set_ylim(ylim_dict[col])

    # shared legend below subplots — collect unique label / color pairs from the last column
    handles_seen: dict[str, Artist] = {}
    for _, _, color, label in pop_cov_data[column_names[-1]]:
        if label not in handles_seen:
            (line,) = axs[-1].plot([], [], color=color, label=label)
            handles_seen[label] = line
    fig.legend(
        handles=list(handles_seen.values()),
        loc="lower center",
        ncol=min(3, len(handles_seen)),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
    )
    fig.suptitle("Population CoV vs time", y=1.01)
    fig.tight_layout()
    fig.savefig(
        fig_savedir / "population_cov_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )

    return fig, axs


def plot_ergodicity_test(
    erg_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
) -> tuple[Figure, list[Axes]]:
    """
    Visualize the ergodicity test by comparing temporal and ensemble CoV.

    For an ergodic system the time-average of an observable computed from a
    single trajectory (= per-crop temporal CoV) should equal the ensemble
    average across all crops at a single snapshot (= mean population CoV).
    Deviations between the two indicate non-ergodic behaviour.

    Each subplot shows a violin plot of the distribution of per-crop temporal
    CoV values (one violin per dataset / flow condition), with a diamond marker
    overlaid at the corresponding mean population CoV.  If the system is
    ergodic the diamond should fall near the centre of the violin.

    Parameters
    ----------
    erg_data
        Mapping from feature column name to a list of
        ``(crop_temporal_cov, mean_pop_cov, color, label)`` tuples — one per
        dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    """
    column_names = list(erg_data.keys())
    n_cols = len(column_names)

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

    for col, ax in zip(column_names, axs, strict=False):
        entries = erg_data[col]
        crop_cov_arrays = [e[0] for e in entries]
        mean_pop_covs = [e[1] for e in entries]
        colors = [e[2] for e in entries]
        labels = [e[3] for e in entries]

        # build long-format DataFrame for seaborn violinplot
        records = [
            (label, float(val))
            for label, cov_arr in zip(labels, crop_cov_arrays, strict=False)
            for val in cov_arr
        ]
        df_viol = pd.DataFrame(records, columns=["condition", "CoV"])
        palette = dict(zip(labels, colors, strict=False))

        # violin plot of per-crop temporal CoV distributions
        sns.violinplot(
            data=df_viol,
            x="condition",
            y="CoV",
            hue="condition",
            legend=False,
            palette=palette,
            inner="quart",
            alpha=0.6,
            linewidth=0.8,
            ax=ax,
            cut=0,
        )

        # diamond marker for mean population (ensemble) CoV
        ax.scatter(
            np.arange(len(labels)),
            mean_pop_covs,
            color=colors,
            marker="D",
            s=60,
            zorder=5,
            edgecolors="k",
            linewidths=0.7,
        )

        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("CoV")
        ax.set_ylim(bottom=0)
        ax.set_title(variable_labels_dict[col])

    # shared legend
    legend_elements = [
        Patch(facecolor="gray", alpha=0.45, label="Per-crop temporal CoV (distribution)"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=8,
            label="Mean population CoV",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=8,
    )
    fig.suptitle(
        "Ergodicity test: individual-crop temporal CoV vs population CoV",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        fig_savedir / "ergodicity_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs


def plot_variance_ratio(
    var_ratio_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the ratio of individual to population variance as a function of time.

    At each timepoint the ratio is the mean per-crop cumulative temporal
    variance divided by the population (cross-sectional) variance.  A shaded
    band shows ± 1 SEM across crops.  A dashed reference line at ratio = 1
    marks the ergodic expectation.

    Each dataset-condition is drawn as a separate line and coloured by shear
    stress regime.

    Parameters
    ----------
    var_ratio_data
        Mapping from feature column name to a list of
        ``(time_values, ratio_mean, ratio_upper, ratio_lower, color, label)``
        tuples — one per dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    """
    column_names = list(var_ratio_data.keys())
    n_cols = len(column_names)

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

    for col, ax in zip(column_names, axs, strict=False):
        for entry in var_ratio_data[col]:
            time_values, ratio_mean, ratio_upper, ratio_lower, color, label = entry
            ax.plot(
                time_values,
                ratio_mean,
                color=color,
                alpha=0.9,
                linewidth=1.2,
            )
            ax.fill_between(
                time_values,
                y1=ratio_upper,
                y2=ratio_lower,
                alpha=0.25,
                color=color,
                label=label,
            )
        # reference line at ratio = 1
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Var$_{\\mathrm{individual}}$ / Var$_{\\mathrm{population}}$")
        ax.set_ylim(0, 1.5)
        ax.set_title(variable_labels_dict[col])

    # shared legend below subplots
    handles_seen: dict[str, Artist] = {}
    for entry in var_ratio_data[column_names[-1]]:
        _, _, _, _, color, label = entry
        if label not in handles_seen:
            handles_seen[label] = Patch(facecolor=color, alpha=0.45, label=label)
    fig.legend(
        handles=list(handles_seen.values()),
        loc="lower center",
        ncol=min(3, len(handles_seen)),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
    )
    fig.suptitle(
        "Individual / population variance ratio vs time",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        fig_savedir / "variance_ratio_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs


def plot_binned_variance_ratio(
    var_ratio_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the ratio of individual to population variance computed within time bins.

    This is the non-cumulative counterpart of :func:`plot_variance_ratio`.
    At each time-bin centre the ratio is the mean per-crop variance *within
    the bin* divided by the population variance within the same bin.  A shaded
    band shows ± 1 SEM across crops.  A dashed reference line at ratio = 1
    marks the ergodic expectation.

    **How to interpret the plot**

    * **Ratio ≈ 1 everywhere** — individual crops fluctuate as much as the
      whole population within each short window → ergodic.
    * **Ratio ≪ 1** — crops occupy narrow, distinct niches in feature space →
      heterogeneous / non-ergodic.
    * **Ratio rising toward 1** — the system is mixing over time.
    * Comparing this binned plot with the cumulative version reveals whether
      ergodicity is driven by local fluctuations (binned ratio ≈ 1) or slow
      drift (cumulative ratio ≈ 1 but binned ratio < 1).

    Parameters
    ----------
    var_ratio_data
        Mapping from feature column name to a list of
        ``(time_values, ratio_mean, ratio_upper, ratio_lower, color, label)``
        tuples — one per dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    """
    column_names = list(var_ratio_data.keys())
    n_cols = len(column_names)

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

    for col, ax in zip(column_names, axs, strict=False):
        for entry in var_ratio_data[col]:
            time_values, ratio_mean, ratio_upper, ratio_lower, color, label = entry
            ax.plot(
                time_values,
                ratio_mean,
                color=color,
                alpha=0.9,
                linewidth=1.2,
            )
            ax.fill_between(
                time_values,
                y1=ratio_upper,
                y2=ratio_lower,
                alpha=0.25,
                color=color,
                label=label,
            )
        # reference line at ratio = 1
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Var$_{\\mathrm{individual}}$ / Var$_{\\mathrm{population}}$ (binned)")
        ax.set_ylim(0, 1.5)
        ax.set_title(variable_labels_dict[col])

    # shared legend below subplots
    handles_seen: dict[str, Artist] = {}
    for entry in var_ratio_data[column_names[-1]]:
        _, _, _, _, color, label = entry
        if label not in handles_seen:
            handles_seen[label] = Patch(facecolor=color, alpha=0.45, label=label)
    fig.legend(
        handles=list(handles_seen.values()),
        loc="lower center",
        ncol=min(3, len(handles_seen)),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
    )
    fig.suptitle(
        "Individual / population variance ratio vs time (binned)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        fig_savedir / "binned_variance_ratio_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs


def plot_mean_feature_vs_time(
    mean_std_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
    filename: str,
    title: str,
    ylabel_suffix: str = "",
) -> tuple[Figure, list[Axes]]:
    """
    Plot population mean ± std of each feature as a function of time.

    Each dataset-condition is drawn as a line (mean) with a shaded band
    (± 1 std) and coloured by shear stress regime.

    Parameters
    ----------
    mean_std_data
        Mapping from feature column name to a list of
        ``(time_values, mean_array, std_array, color, label)`` tuples — one
        per dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    filename
        Filename (without directory) for the saved figure.
    title
        Figure suptitle.
    ylabel_suffix
        Optional suffix appended to each y-axis label (e.g. " (scaled)").
    """
    column_names = list(mean_std_data.keys())
    n_cols = len(column_names)

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

    for col, ax in zip(column_names, axs, strict=False):
        for entry in mean_std_data[col]:
            time_values, mean_arr, std_arr, color, label = entry
            ax.plot(
                time_values,
                mean_arr,
                color=color,
                alpha=0.9,
                linewidth=1.2,
            )
            ax.fill_between(
                time_values,
                y1=mean_arr + std_arr,
                y2=mean_arr - std_arr,
                alpha=0.25,
                color=color,
                label=label,
                edgecolor="none",
            )
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"{variable_labels_dict[col]}{ylabel_suffix}")
        ax.set_title(variable_labels_dict[col])

    # shared legend below subplots
    handles_seen: dict[str, Artist] = {}
    for entry in mean_std_data[column_names[-1]]:
        _, _, _, color, label = entry
        if label not in handles_seen:
            handles_seen[label] = Patch(facecolor=color, alpha=0.45, label=label)
    fig.legend(
        handles=list(handles_seen.values()),
        loc="lower center",
        ncol=min(3, len(handles_seen)),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
    )
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(
        fig_savedir / filename,
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs
