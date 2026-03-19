import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binned_statistic_2d

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import check_required_columns_in_dataframe
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP

logger = logging.getLogger(__name__)


def plot_optical_flow_feature_distribution(
    df: pd.DataFrame,
    optical_flow_feature: str,
    datasets: list[str],
    plot_label: str,
    output_dir: Path,
    binwidth: float = 0.02,
    bins: int = 50,
    kde: bool = True,
    figsize: tuple[float, float] = (4, 2.5),
) -> None:
    """
    Plot an optical-flow feature histogram per dataset on a shared axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing a ``"dataset"`` column and the column named by
        *optical_flow_feature*.
    optical_flow_feature : str
        Column name of the optical-flow feature to plot.
    datasets : list[str]
        Dataset identifiers to include. Each dataset is plotted as a separate
        histogram with its own colour and shear-stress label.
    output_dir : Path
        Directory where the figure is saved.
    binwidth : float, default=0.02
        Width of each histogram bin passed to :func:`seaborn.histplot`.
    bins : int, default=50
        Number of histogram bins passed to :func:`seaborn.histplot`.
    kde : bool, default=True
        Whether to overlay a kernel-density estimate on the histogram.
    figsize : tuple[float, float], default=(4, 2.5)
        Width and height of the figure in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for dataset in datasets:
        color = get_dataset_color(dataset)

        df_of_subset = df[df["dataset"] == dataset]
        sns.histplot(
            df_of_subset[optical_flow_feature],
            bins=bins,
            kde=kde,
            label=plot_label,
            binwidth=binwidth,
            ax=ax,
            color=color,
        )

    ax.set_xlabel(optical_flow_feature)
    ax.set_ylabel("Count")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    plt.show()
    save_plot_to_path(fig, output_dir, f"{optical_flow_feature}_dist_{'_'.join(datasets)}.png")
    plt.close(fig)


def plot_scatter_and_binned_heatmap(
    df: pd.DataFrame,
    dataset_name: str,
    x_col: str,
    y_col: str,
    color_col: str,
    colormap: str = MIGRATION_COHERENCE_COLORMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    x_bin_size: float = 0.25,
    y_bin_size: float = 0.25,
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """Plot scatter (left) and binned mean heatmap (right) side by side.

    The left panel shows a per-point scatter colored by *color_col* and the
    right panel shows the mean of *color_col* within 2-D bins.

    Parameters
    ----------
    df
        Dataframe containing columns *x_col*, *y_col*, *color_col*, and
        ``"dataset"``.
    dataset_name
        Dataset identifier used to filter rows and label the figure title with
        the corresponding shear-stress condition.
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
        required_columns=[x_col, y_col, color_col, ColumnName.DATASET],
    )
    cmap = plt.get_cmap(colormap)
    df_plot = df[(df[ColumnName.DATASET] == dataset_name) & df[color_col].notna()]
    x = df_plot[x_col].to_numpy()
    y = df_plot[y_col].to_numpy()
    z = df_plot[color_col].to_numpy()

    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left: scatter plot
    axs[0].scatter(x, y, c=z, cmap=cmap, s=5, vmin=vmin, vmax=vmax)
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

    dataset_config = load_dataset_config(dataset_name)
    flow_conditions = dataset_config.flow_conditions
    shear_stress_values = [fc.shear_stress for fc in flow_conditions]
    shear_stress_label = "-".join(f"{v:g}" for v in shear_stress_values)
    title = f"{dataset_name}, {shear_stress_label} dyn/cm^2"

    plt.suptitle(title)
    plt.tight_layout()
    return fig, axs
