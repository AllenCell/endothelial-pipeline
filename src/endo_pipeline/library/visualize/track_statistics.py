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
    kernel_name: str,
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
