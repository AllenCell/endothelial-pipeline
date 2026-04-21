"""Module for visualizing outputs of long-time-scale statistics analyses for time series data."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    get_kernel_density_estimate_from_histogram,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins


def plot_kde(
    axes: plt.Axes,
    x_eval: np.ndarray,
    kde_values: np.ndarray,
    kde_line_style: str = "-",
    kde_color: str = "k",
    kde_label: str | None = None,
    kde_linewidth: float = 2.0,
    axes_title: str | None = None,
    axes_xlimits: tuple[float, float] | None = None,
    axes_xlabel: str | None = None,
    axes_ylabel: str | None = None,
) -> None:
    """Add a pre-computed kernel density estimate (KDE) to existing axes.

    Parameters
    ----------
    axes
        The matplotlib Axes object to plot on.
    x_eval
        1D array of x-values at which the KDE was evaluated.
    kde_values
        1D array of KDE values at each point in ``x_eval``, as returned by
        :func:`compute_interpolated_kde_spline`.
    kde_line_style
        The line style to use for the KDE plot.
    kde_color
        The color to use for the KDE plot.
    kde_label
        The label to use for the KDE plot in the legend (set to None to omit
        from legend).
    kde_linewidth
        The line width to use for the KDE plot.
    axes_title
        The title to set for the axes (set to None to omit).
    axes_xlimits
        The x-axis limits to set for the plot (set to None to auto-scale).
    axes_xlabel
        The label to set for the x-axis (set to None to omit).
    axes_ylabel
        The label to set for the y-axis (set to None to omit).

    """
    axes.plot(
        x_eval,
        kde_values,
        color=kde_color,
        linewidth=kde_linewidth,
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


def compute_kde_on_bins(
    data: np.ndarray,
    bin_width: float,
    kernel_name: Literal["gaussian", "epanechnikov", "periodic"],
    kernel_bandwidth: float,
    kernel_period: float | None,
    bins: np.ndarray | None = None,
    pad_bins: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a kernel density estimate (KDE) on the native histogram bin centers.

    Unlike :func:`compute_interpolated_kde_spline`, this function returns KDE
    values on the coarse bin-center grid without any spline smoothing.  This
    is the right representation for accumulating bootstrap samples: average the
    raw KDE values first, then apply spline smoothing only once at plot time
    via :func:`smooth_kde_with_spline`.

    Parameters
    ----------
    data
        1D array of data points to estimate the density for.
    bin_width
        The width of the histogram bins used to compute the KDE.
    kernel_name
        The name of the kernel to use for the KDE.
    kernel_bandwidth
        The bandwidth parameter for the kernel density estimate.
    kernel_period
        The period for periodic kernels (pass None for non-periodic kernels).
    bins
        Pre-computed bin edges (1D array) to use instead of deriving them from
        ``data``.  Pass this to ensure all bootstrap samples share the same
        bin grid so their KDE arrays can be stacked and averaged.  When
        ``None``, bin edges are derived from ``data``.
    pad_bins
        Amount to pad histogram bins on either side of the data range.
        Ignored when ``bins`` is provided.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bin_centers, kde_values)`` — both 1D arrays of the same length.

    """
    if bins is not None:
        computed_bins = [bins]
        centers = [(bins[:-1] + bins[1:]) / 2]
    else:
        computed_bins, centers = get_bins(bin_widths=(bin_width,), data=data, pad=pad_bins)
    hist = np.histogram(data, bins=computed_bins[0], density=True)[0]
    kernel = KramersMoyalKernel(
        name=kernel_name,
        bandwidth=kernel_bandwidth,
        period=kernel_period,
    )
    hist_kde = get_kernel_density_estimate_from_histogram(hist, bins=computed_bins, kernel=kernel)
    return centers[0], hist_kde


def smooth_kde_with_spline(
    bin_centers: np.ndarray,
    kde_values: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """Fit a cubic spline to a KDE and evaluate it on a fine grid.

    Intended to be called at *plot time* on KDE values that were first computed
    (and optionally averaged / CI-bounded) on a coarse bin-center grid via
    :func:`compute_kde_on_bins`.

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
    np.ndarray
        KDE values evaluated at each point in ``x_eval``.  Positions where
        ``kde_values`` is not finite are excluded from the spline fit.
        Returns all-NaN if fewer than four finite values exist.

    """
    finite_mask = np.isfinite(kde_values)
    if finite_mask.sum() < 4:
        return np.full(len(x_eval), np.nan)
    knot_x = bin_centers[finite_mask]
    spline = make_interp_spline(knot_x, kde_values[finite_mask], k=3)
    return spline(x_eval)


def compute_interpolated_kde_spline(
    data: np.ndarray,
    x_eval: np.ndarray,
    bin_width: float,
    kernel_name: Literal["gaussian", "epanechnikov", "periodic"],
    kernel_bandwidth: float,
    kernel_period: float | None,
    pad_bins: float = 0.0,
) -> np.ndarray:
    """Compute a kernel density estimate (KDE) for data, evaluated at fixed x_eval points.

    Parameters
    ----------
    data
        1D array of data points to estimate the density for.
    x_eval
        1D array of x-values at which to evaluate the KDE.
    bin_width
        The width of the histogram bins used to compute the KDE.
    kernel_name
        The name of the kernel to use for the KDE.
    kernel_bandwidth
        The bandwidth parameter for the kernel density estimate.
    kernel_period
        The period for periodic kernels (pass None for non-periodic kernels).
    pad_bins
        Amount to pad histogram bins on either side of the data range.

    Returns
    -------
    np.ndarray
        KDE values evaluated at each point in ``x_eval``. Points outside the
        data range or where the spline cannot be computed are returned as NaN.

    """
    bin_centers, kde_values = compute_kde_on_bins(
        data=data,
        bin_width=bin_width,
        kernel_name=kernel_name,
        kernel_bandwidth=kernel_bandwidth,
        kernel_period=kernel_period,
        pad_bins=pad_bins,
    )
    return smooth_kde_with_spline(bin_centers, kde_values, x_eval)
