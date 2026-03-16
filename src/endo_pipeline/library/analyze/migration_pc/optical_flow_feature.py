import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binned_statistic_2d

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.workflow_defaults import OPTICAL_FLOW_BASE_FEATURES

logger = logging.getLogger(__name__)


def add_optical_flow_features(
    df: pd.DataFrame,
    datasets: list[str],
    optical_flow_manifest_name: str = "optical_flow_bf",
    optical_flow_feature_columns: list[str] = OPTICAL_FLOW_BASE_FEATURES,
) -> pd.DataFrame:
    """Load optical-flow features and merge them with a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
            Input dataframe containing rows to enrich with optical-flow features.
    datasets : list[str]
            Dataset names to process.
    optical_flow_manifest_name : str, default="optical_flow_bf"
            Name of the dataframe manifest containing optical-flow feature tables.
    optical_flow_feature_columns : list[str], default=OPTICAL_FLOW_BASE_FEATURES
            List of optical-flow feature column names to merge into the input dataframe.

    Returns
    -------
    pandas.DataFrame
            Concatenated dataframe with optical-flow features merged in.
    """

    merge_columns = ["dataset", "position", "frame_number", "start_x", "start_y"]
    dataframe_manifest_optical_flow = load_dataframe_manifest(optical_flow_manifest_name)

    merged_dfs = []
    for dataset_name in datasets:
        logger.info("Adding optical flow features for dataset: %s", dataset_name)

        df_dataset = df[df["dataset"] == dataset_name]

        optical_flow_location = get_dataframe_location_for_dataset(
            dataframe_manifest_optical_flow, dataset_name
        )
        df_optical_flow = load_dataframe(optical_flow_location)
        df_optical_flow = df_optical_flow[merge_columns + optical_flow_feature_columns]

        df_merged = df_dataset.merge(
            df_optical_flow,
            on=merge_columns,
            how="left",
        )
        merged_dfs.append(df_merged)

    return pd.concat(merged_dfs, ignore_index=True)


def plot_optical_flow_feature_distribution(
    df: pd.DataFrame,
    optical_flow_feature: str,
    datasets: list[str],
    output_dir: Path,
    binwidth: float = 0.02,
    bins: int = 50,
    kde: bool = True,
    figsize: tuple[float, float] = (4, 2.5),
) -> None:
    """Plot an optical-flow feature histogram per dataset on a shared axis.

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

        dataset_config = load_dataset_config(dataset)
        flow_conditions = dataset_config.flow_conditions
        shear_stress_values = [flow_condition.shear_stress for flow_condition in flow_conditions]
        shear_stress_label = "-".join(f"{value:g}" for value in shear_stress_values)
        df_of_subset = df[df["dataset"] == dataset]
        sns.histplot(
            df_of_subset[optical_flow_feature],
            bins=bins,
            kde=kde,
            label=f"{dataset}, shear={shear_stress_label}",
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
    output_dir: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    x_bin_size: float = 0.25,
    y_bin_size: float = 0.25,
) -> None:
    """Plot scatter (left) and binned mean heatmap (right) side by side.

    The left panel shows a per-point scatter coloured by *color_col* and the
    right panel shows the mean of *color_col* within 2-D bins.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing columns *x_col*, *y_col*, *color_col*, and
        ``"dataset"``.
    dataset_name : str
        Dataset identifier used to filter rows and label the figure title
        with the corresponding shear-stress condition.
    x_col : str
        Column name for the x-axis of both panels.
    y_col : str
        Column name for the y-axis of both panels.
    color_col : str
        Column name whose values are mapped to colour in the scatter and
        averaged per bin in the heatmap.
    output_dir : Path
        Directory where the figure will be saved.
    vmin : float or None, default=None
        Lower bound of the colour scale. If ``None``, derived from the data.
    vmax : float or None, default=None
        Upper bound of the colour scale. If ``None``, derived from the data.
    x_bin_size : float, default=0.25
        Bin width along the x-axis for the heatmap.
    y_bin_size : float, default=0.25
        Bin width along the y-axis for the heatmap.
    """
    cmap = plt.get_cmap("cool")
    df_plot = df[(df["dataset"] == dataset_name) & df[color_col].notna()]
    x = df_plot[x_col].values
    y = df_plot[y_col].values
    z = df_plot[color_col].values

    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: scatter plot
    ax1.scatter(x, y, c=z, cmap=cmap, s=5, vmin=vmin, vmax=vmax)
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)

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
    im = ax2.pcolormesh(
        x_edges,
        y_edges,
        stat.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    fig.colorbar(im, ax=ax2, label=color_col)
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)

    dataset_config = load_dataset_config(dataset_name)
    flow_conditions = dataset_config.flow_conditions
    shear_stress_values = [fc.shear_stress for fc in flow_conditions]
    shear_stress_label = "-".join(f"{v:g}" for v in shear_stress_values)
    title = f"{dataset_name}, {shear_stress_label} dyn/cm^2"

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    save_plot_to_path(fig, output_dir, f"{dataset_name}_{x_col}_vs_{y_col}_colored_by_{color_col}")
    plt.close(fig)
