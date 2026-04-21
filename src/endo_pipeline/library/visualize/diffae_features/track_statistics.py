"""Module for visualizing outputs of long-time-scale statistics analyses for time series data."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def smooth_kde_with_spline(
    bin_centers: np.ndarray,
    kde_values: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Fit a cubic spline to a KDE and evaluate it on a fine grid.

    Intended to be called at *plot time* on KDE values that were first computed
    (and optionally averaged / CI-bounded) on a coarse bin-center grid via
    :func:`compute_kde_on_bins`.

    **NaN handling**:

    Only the finite values of `kde_values` are used for fitting the spline. If
    fewer than four finite values exist, the function returns an array of NaNs.

    Parameters
    ----------
    bin_centers
        1D array of bin-center x-values (the coarse grid).
    kde_values
        1D array of KDE values at each bin center.
    x_eval
        1D array of x-values at which to evaluate the smoothed spline.

    Returns
    -------
    :
        KDE values evaluated at each point in `x_eval`.

    """
    finite_mask = np.isfinite(kde_values)
    if finite_mask.sum() < 4:
        return np.full(len(x_eval), np.nan)
    knot_x = bin_centers[finite_mask]
    spline = make_interp_spline(knot_x, kde_values[finite_mask], k=3)
    return spline(x_eval)


def plot_kde_for_track_statistics(
    ax: plt.Axes,
    kde_values: np.ndarray,
    bin_centers: np.ndarray,
    x_eval: np.ndarray,
    kde_ci_lower: np.ndarray | None = None,
    kde_ci_upper: np.ndarray | None = None,
    axes_title: str | None = None,
    axes_xlabel: str | None = None,
    axes_ylabel: str | None = None,
    kde_line_kwargs: dict | None = None,
    ci_line_kwargs: dict | None = None,
) -> None:
    """
    Plot KDE for visualization of track statistic comparison results.

    Parameters
    ----------
    ax
        The matplotlib Axes object to plot on.
    kde_values
        1D array of KDE values corresponding to the bin centers.
    bin_centers
        1D array of bin center values corresponding to the KDE values.
    x_eval
        1D array of x values to evaluate the smoothed KDE on for plotting.
    kde_ci_lower
        Optional, 1D array of lower confidence interval values for the KDE,
        corresponding to the bin centers.
    kde_ci_upper
        Optional, 1D array of upper confidence interval values for the KDE,
        corresponding to the bin centers.
    axes_title
        Optional, title to set for the axes.
    axes_xlabel
        Optional, label to set for the x-axis.
    axes_ylabel
        Optional, label to set for the y-axis.
    kde_line_kwargs
        Optional, dictionary of keyword arguments to pass to the plt.plot call
        for the KDE line (e.g, color, linestyle, linewidth).
    ci_line_kwargs
        Optional, dictionary of keyword arguments to pass to the
        plt.fill_between call for the confidence interval shading (e.g, color,
        alpha).

    """
    plt.style.use("endo_pipeline.figure")

    # Smooth the KDE values using spline interpolation for plotting
    kde_smooth = smooth_kde_with_spline(
        bin_centers=bin_centers,
        kde_values=kde_values,
        x_eval=x_eval,
    )
    kde_line = ax.plot(
        x_eval,
        kde_smooth,
        **(kde_line_kwargs or {}),
    )

    # If confidence interval values are provided, smooth them and add shaded
    # area to the plot
    if kde_ci_lower is not None and kde_ci_upper is not None:
        kde_color = kde_line[0].get_color()
        if ci_line_kwargs is None:
            ci_line_kwargs = {"alpha": 0.25, "color": kde_color}
        elif "color" not in ci_line_kwargs:
            ci_line_kwargs["color"] = kde_color

        kde_ci_lower_smooth = smooth_kde_with_spline(
            bin_centers=bin_centers,
            kde_values=kde_ci_lower,
            x_eval=x_eval,
        )
        kde_ci_upper_smooth = smooth_kde_with_spline(
            bin_centers=bin_centers,
            kde_values=kde_ci_upper,
            x_eval=x_eval,
        )
        ax.fill_between(
            x_eval,
            kde_ci_lower_smooth,
            kde_ci_upper_smooth,
            **ci_line_kwargs,
        )

    if axes_title is not None:
        ax.set_title(axes_title)
    if axes_xlabel is not None:
        ax.set_xlabel(axes_xlabel)
    if axes_ylabel is not None:
        ax.set_ylabel(axes_ylabel)
