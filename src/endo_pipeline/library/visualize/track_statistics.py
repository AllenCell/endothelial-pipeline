import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


def plot_histogram_with_kde(
    ax: plt.Axes,
    histogram: np.ndarray,
    histogram_bins: np.ndarray,
    histogram_kde: np.ndarray,
    kde_points: np.ndarray,
    histogram_color: str = "blue",
    histogram_alpha: float = 0.5,
    histogram_edge_color: str = "black",
    histogram_edge_alpha: float = 1.0,
    kde_color: str = "k",
    kde_linewidth: float = 1.5,
    kde_linestyle: str = "-",
) -> None:
    """
    Plot histogram with KDE overlaid.

    Parameters
    ----------
    ax
        Matplotlib axes to plot on.
    histogram
        Histogram values (e.g., from `numpy.histogram`).
    histogram_bins
        Histogram bin edges (e.g., from `numpy.histogram`).
    histogram_kde
        KDE values corresponding to `kde_points`.
    kde_points
        Points at which KDE is evaluated.
    histogram_color
        Color for histogram bars.
    histogram_alpha
        Alpha (transparency) for histogram bars.
    histogram_edge_color
        Color for histogram bar edges.
    histogram_edge_alpha
        Alpha (transparency) for histogram bar edges.
    kde_color
        Color for KDE line.
    kde_linewidth
        Line width for KDE line.
    kde_linestyle
        Line style for KDE line (e.g., "-", "--", "-.", ":").
    """
    # plot histogram as bars with semi-transparency
    ax.bar(
        histogram_bins[:-1],
        histogram,
        width=np.diff(histogram_bins),
        color=(*to_rgb(histogram_color), histogram_alpha),
        edgecolor=(*to_rgb(histogram_edge_color), histogram_edge_alpha),
        align="edge",
    )
    ax.plot(
        kde_points,
        histogram_kde,
        color=kde_color,
        linewidth=kde_linewidth,
        linestyle=kde_linestyle,
    )


def plot_histogram_and_kde_with_confidence_interval(
    ax: plt.Axes,
    histogram: np.ndarray,
    histogram_bins: np.ndarray,
    histogram_confidence_interval: tuple[np.ndarray, np.ndarray],
    histogram_kde: np.ndarray,
    kde_points: np.ndarray,
    kde_confidence_interval: tuple[np.ndarray, np.ndarray],
    histogram_color: str = "blue",
    histogram_alpha: float = 0.5,
    histogram_edge_color: str = "black",
    histogram_edge_alpha: float = 1.0,
    kde_color: str = "k",
    kde_linewidth: float = 1.5,
    kde_linestyle: str = "-",
    kde_confidence_interval_alpha: float = 0.3,
) -> None:
    """
    Plot histogram with KDE overlaid, along with shaded confidence interval for
    KDE.

    Parameters
    ----------
    ax
        Matplotlib axes to plot on.
    histogram
        Histogram values (e.g., from `numpy.histogram`).
    histogram_bins
        Histogram bin edges (e.g., from `numpy.histogram`).
    histogram_confidence_interval
        Tuple of (lower_bound, upper_bound) arrays for histogram confidence
        interval.
    histogram_kde
        KDE values corresponding to `kde_points`.
    kde_points
        Points at which KDE is evaluated.
    kde_confidence_interval
        Tuple of (lower_bound, upper_bound) arrays for KDE confidence interval.
    histogram_color
        Color for histogram bars.
    histogram_alpha
        Alpha (transparency) for histogram bars.
    histogram_edge_color
        Color for histogram bar edges.
    histogram_edge_alpha
        Alpha (transparency) for histogram bar edges.
    kde_color
        Color for KDE line.
    kde_linewidth
        Line width for KDE line.
    kde_linestyle
        Line style for KDE line (e.g., "-", "--", "-.", ":").
    kde_confidence_interval_alpha
        Alpha (transparency) for confidence interval shading on KDE plot.
    """
    # plot histogram and kde as usual
    plot_histogram_with_kde(
        ax,
        histogram=histogram,
        histogram_bins=histogram_bins,
        histogram_kde=histogram_kde,
        kde_points=kde_points,
        histogram_color=histogram_color,
        histogram_alpha=histogram_alpha,
        histogram_edge_color=histogram_edge_color,
        histogram_edge_alpha=histogram_edge_alpha,
        kde_color=kde_color,
        kde_linewidth=kde_linewidth,
        kde_linestyle=kde_linestyle,
    )

    # add confidence interval for histogram as error bars
    histogram_bin_centers = (histogram_bins[:-1] + histogram_bins[1:]) / 2
    histogram_error = np.array(
        [
            histogram_confidence_interval[1] - histogram,
            histogram - histogram_confidence_interval[0],
        ]
    )
    ax.errorbar(
        histogram_bin_centers,
        histogram,
        yerr=histogram_error,
        fmt="none",
        ecolor=histogram_edge_color,
        capsize=3,
    )

    # plot KDE line
    ax.plot(
        kde_points,
        histogram_kde,
        color=kde_color,
        linewidth=kde_linewidth,
        linestyle=kde_linestyle,
    )
    # add shaded confidence interval for KDE
    ax.fill_between(
        kde_points,
        kde_confidence_interval[0],
        kde_confidence_interval[1],
        color=kde_color,
        alpha=kde_confidence_interval_alpha,
    )
