"""Module for visualizing outputs of long-time-scale statistics analyses for time series data."""

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.library.analyze.numerics.temporal_stats import smooth_kde_with_spline


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
    ci_line_kwargs: dict | None = {"alpha": 0.25},
) -> plt.Axes:
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

    Returns
    -------
    :
        The matplotlib Axes object with the KDE plot added.

    """
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
        if "color" not in ci_line_kwargs:
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
            color=kde_line_kwargs.get("color", "k"),
            **ci_line_kwargs,
        )

    if axes_title is not None:
        ax.set_title(axes_title)
    if axes_xlabel is not None:
        ax.set_xlabel(axes_xlabel)
    if axes_ylabel is not None:
        ax.set_ylabel(axes_ylabel)

    return ax
