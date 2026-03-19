import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from endo_pipeline.library.analyze.diffae_dataframe_utils import check_required_columns_in_dataframe
from endo_pipeline.settings.migration_coherence import (
    MIGRATION_COHERENCE_COLORMAP,
    MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
)

logger = logging.getLogger(__name__)


def plot_scatter_and_binned_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    colormap: str = MIGRATION_COHERENCE_COLORMAP,
    vmin: float | None = 0,
    vmax: float | None = 1,
    x_bin_size: float = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
    y_bin_size: float = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
    figsize: tuple[float, float] = (10, 5),
    scatter_point_size: float = 5,
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """
    Plot scatter and binned mean heatmap over the same x and y columns, colored
    by a specified feature column.

    **Dataframe columns and plot description**

    The input dataframe must contain the columns specified in `x_col`, `y_col`,
    and `color_col`.

    The left panel of the plot is a per-point scatter of `x_col` vs `y_col`
    colored by `color_col`. The right panel shows the mean of `color_col` within
    2-D bins of `x_col` and `y_col`, where the bin sizes are specified by
    `x_bin_size` and `y_bin_size`.

    Both panels share the same x and y limits, which are determined by the range
    of the data in `x_col` and `y_col`.

    The color scale for both panels is determined by the range of values in
    `color_col`.

    Parameters
    ----------
    df
        Dataframe containing columns for plotting.
    x_col
        Column name for the x-axis of both panels.
    y_col
        Column name for the y-axis of both panels.
    color_col
        Column name whose values are mapped to color in the scatter and averaged
        per bin in the heatmap.
    colormap
        Name of the matplotlib colormap to use for coloring points and bins
        based on *color_col* values.
    vmin
        Lower bound of the color scale. If ``None``, derived from the data.
    vmax
        Upper bound of the color scale. If ``None``, derived from the data.
    x_bin_size
        Bin width along the x-axis for the heatmap.
    y_bin_size
        Bin width along the y-axis for the heatmap.
    """

    check_required_columns_in_dataframe(
        df,
        required_columns=[x_col, y_col, color_col],
    )
    cmap = plt.get_cmap(colormap)
    df_plot = df[df[color_col].notna()]
    x = df_plot[x_col].to_numpy()
    y = df_plot[y_col].to_numpy()
    z = df_plot[color_col].to_numpy()

    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Left: scatter plot
    axs[0].scatter(x, y, c=z, cmap=cmap, s=scatter_point_size, vmin=vmin, vmax=vmax)
    axs[0].set_xlabel(x_col)
    axs[0].set_ylabel(y_col)

    # Right: binned heatmap
    x_bins = np.arange(x.min(), x.max() + x_bin_size, x_bin_size)
    y_bins = np.arange(y.min(), y.max() + y_bin_size, y_bin_size)
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        x,
        y,
        z,
        statistic="mean",
        bins=[x_bins, y_bins],
    )
    im = axs[1].pcolormesh(
        x_edges,
        y_edges,
        stat.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs[1].set_xlim(axs[0].get_xlim())
    axs[1].set_ylim(axs[0].get_ylim())
    axs[1].set_xlabel(x_col)
    axs[1].set_ylabel(y_col)
    # add colorbar for the heatmap without resizing the main axes
    cax = axs[1].inset_axes([1.05, 0, 0.05, 1])
    fig.colorbar(im, cax=cax, label=color_col)

    return fig, axs
