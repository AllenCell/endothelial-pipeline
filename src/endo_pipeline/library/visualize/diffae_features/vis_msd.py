"""Visualization methods for MSD (mean squared displacement) analysis."""

import matplotlib.pyplot as plt
import numpy as np


def plot_msd_with_exponential_fit(
    msd_vals: np.ndarray,
    lags: np.ndarray,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fig_title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot mean squared displacement (MSD) values against time lags with an exponential fit.

    Plot is generated on a log-log scale, and an exponential fit is overlaid to
    show the relationship between MSD and time lags.

    Parameters
    ----------
    msd_vals
        An array of mean squared displacement values corresponding to each time lag.
    lags
        An array of time lags corresponding to the MSD values.
    xlabel
        Optional, label for the x-axis.
    ylabel
        Optional, label for the y-axis.
    fig_title
        Optional, title for the plot.
    xlim
        Optional, tuple specifying the limits for the x-axis (min, max).
    ylim
        Optional, tuple specifying the limits for the y-axis (min, max).


    Returns
    -------
    :
        The matplotlib Figure object for the generated plot.

    """
    where_finite = np.isfinite(msd_vals)
    linear_fit, res, _, _, _ = np.polyfit(
        np.log(lags[where_finite]),
        np.log(msd_vals[where_finite]),
        1,
        full=True,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(lags * 5, msd_vals, "k-", marker="o")
    ax.loglog(
        lags * 5,
        np.exp(linear_fit[1]) * (lags ** linear_fit[0]),
        "b--",
        label=f"MSD ~ $\\Delta t^{{{linear_fit[0]:.2f}}}$ (R$^2$ = {1-res[0]:.2f}):",
    )
    ax.legend()

    if fig_title is not None:
        ax.set_title(fig_title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig
