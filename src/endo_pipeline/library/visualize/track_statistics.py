"""Module for visualizing outputs of long-time-scale statistics analyses for time series data."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.interpolate import make_interp_spline

from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    get_kernel_density_estimate_from_histogram,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins


def plot_histogram_and_kde(
    axes: plt.Axes,
    data: np.ndarray,
    bin_width: float,
    kernel_name: Literal["gaussian", "epanechnikov", "periodic"],
    kernel_bandwidth: float,
    kernel_period: float | None,
    hist_color: str = "blue",
    hist_alpha: float = 0.5,
    kde_line_style: str = "-",
    kde_color: str = "k",
    kde_label: str | None = None,
    pad_bins: float = 0.0,
    axes_title: str | None = None,
    axes_xlimits: tuple[float, float] | None = None,
    axes_xlabel: str | None = None,
    axes_ylabel: str | None = None,
) -> None:
    """Add a histogram of input data with an overlaid kernel density estimate (KDE) to existing axes.

    Parameters
    ----------
    axes
        The matplotlib Axes object to plot on.
    data
        1D array of data points to plot the histogram and KDE for.
    bin_width
        The width of the bins to use for the histogram.
    kernel_name
        The name of the kernel to use for the KDE.
    kernel_bandwidth
        The bandwidth parameter for the kernel density estimate.
    kernel_period
        The period parameter for the kernel density estimate (only used for
        periodic kernels).
    hist_color
        The color to use for the histogram bars.
    hist_alpha
        The alpha (transparency) value to use for the histogram bars.
    kde_line_style
        The line style to use for the KDE plot.
    kde_color
        The color to use for the KDE plot.
    kde_label
        The label to use for the KDE plot in the legend (set to None to omit
        from legend).
    pad_bins
        The amount to pad the histogram bins on either side of the data range.
    axes_title
        The title to set for the axes (set to None to omit).
    axes_xlimits
        The x-axis limits to set for the plot (set to None to auto-scale).
    axes_xlabel
        The label to set for the x-axis (set to None to omit).
    axes_ylabel
        The label to set for the y-axis (set to None to omit).

    """
    # get histogram of the column average using bin widths of 0.1,
    # adjusting x-axis limits based on bin limits for the column
    bins, centers = get_bins(bin_widths=(bin_width,), data=data, pad=pad_bins)
    hist = np.histogram(data, bins=bins[0], density=True)[0]
    kernel = KramersMoyalKernel(
        name=kernel_name,
        bandwidth=kernel_bandwidth,
        period=kernel_period,
    )
    hist_kde = get_kernel_density_estimate_from_histogram(hist, bins=bins, kernel=kernel)
    # interpolate between histogram centers for smoother KDE plot
    interp_centers = np.linspace(bins[0][0], bins[0][-1], 2000)
    spline = make_interp_spline(centers[0], hist_kde, k=3)  # k=3 for cubic spline
    hist_kde_smooth = spline(interp_centers)

    # plot histogram of the column variance with KDE overlaid
    axes.bar(
        bins[0][:-1],
        hist,
        width=np.diff(bins[0]),
        color=(*to_rgb(hist_color), hist_alpha),
        edgecolor=(*to_rgb("k"), 1.0),
        align="edge",
    )
    axes.plot(
        interp_centers,
        hist_kde_smooth,
        color=kde_color,
        linewidth=1.5,
        linestyle=kde_line_style,
        label=kde_label,
    )
    if axes_title is not None:
        axes.set_title(axes_title)
    if axes_xlimits is not None:
        axes.set_xlim(axes_xlimits)
    if axes_xlabel is not None:
        axes.set_xlabel(axes_xlabel)
    if axes_ylabel is not None:
        axes.set_ylabel(axes_ylabel)
    if kde_label is not None:
        axes.legend(loc="upper right")
