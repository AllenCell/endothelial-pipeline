import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binned_statistic_2d, binned_statistic_dd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import check_required_columns_in_dataframe
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.flow_field_dataframes import (
    STABILITY_COLOR_DICT,
    STABILITY_COLUMN_NAME,
    STABILITY_MARKER_DICT,
)
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP

logger = logging.getLogger(__name__)


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


def plot_3d_scatter_or_binned(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str,
    dataset_name: str,
    df_fp: pd.DataFrame,
    output_dir: Path,
    binned: bool = False,
    bin_size_xyz: tuple[float, float, float] = (0.25, 0.25, 0.25),
    cmap: str = "cool",
    vmin: float = 0,
    vmax: float = 1,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 3D scatter or 3D binned heatmap with optional fixed-point overlay.

    Parameters
    ----------
    df
        Dataframe containing columns *x_col*, *y_col*, *z_col*, and *color_col*.
    x_col, y_col, z_col
        Column names for the three spatial axes.
    color_col
        Column name whose values are mapped to color.
    dataset_name
        Dataset identifier for the title.
    df_fp
        Fixed-points dataframe. Fixed points are overlaid with stability-specific markers and colors.
    binned
        If ``False`` (default), plot every point as a scatter.
        If ``True``, bin the data in 3D and show the mean of
        *color_col* per bin as colored squares.
    bin_size_xyz
        Bin widths ``(x_bin, y_bin, z_bin)`` along each axis
        (only used when ``mode="binned"``).
    cmap
        Matplotlib colormap name.
    vmin, vmax
        Color-scale limits.
    """
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    z = df[z_col].to_numpy()
    c = df[color_col].to_numpy()

    x_bin_size, y_bin_size, z_bin_size = bin_size_xyz

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    if not binned:
        sc = ax.scatter(
            x,
            y,
            z,
            c=c,
            cmap=cmap,
            s=5,
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,
            zorder=1,
        )
        cbar_label = color_col
    else:
        # Bin the data in 3D and compute the mean of c per bin
        x_bins = np.arange(x.min(), x.max() + x_bin_size, x_bin_size)
        y_bins = np.arange(y.min(), y.max() + y_bin_size, y_bin_size)
        z_bins = np.arange(z.min(), z.max() + z_bin_size, z_bin_size)

        stat, bin_edges, _ = binned_statistic_dd(
            sample=np.column_stack([x, y, z]),
            values=c,
            statistic="mean",
            bins=[x_bins, y_bins, z_bins],
        )

        bcx = 0.5 * (bin_edges[0][:-1] + bin_edges[0][1:])
        bcy = 0.5 * (bin_edges[1][:-1] + bin_edges[1][1:])
        bcz = 0.5 * (bin_edges[2][:-1] + bin_edges[2][1:])
        gx, gy, gz = np.meshgrid(bcx, bcy, bcz, indexing="ij")

        mask = ~np.isnan(stat.ravel())
        sc = ax.scatter(
            gx.ravel()[mask],
            gy.ravel()[mask],
            gz.ravel()[mask],
            c=stat.ravel()[mask],
            cmap=cmap,
            s=80,
            marker="s",
            vmin=vmin,
            vmax=vmax,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.8,
            zorder=1,
        )
        cbar_label = f"mean {color_col}"

    # Overlay fixed points
    from matplotlib.lines import Line2D

    legend_handles = []
    for _, row in df_fp.iterrows():
        stability = row[STABILITY_COLUMN_NAME]
        mk = STABILITY_MARKER_DICT.get(stability, "o")
        clr = STABILITY_COLOR_DICT.get(stability, "gray")
        theta, r, rho = row[x_col], row[y_col], row[z_col]
        mean_val = row.get(f"mean_{color_col}", float("nan"))
        ax.scatter(
            [theta],
            [r],
            [rho],
            marker=mk,
            color=clr,
            edgecolor="black",
            linewidths=1.5,
            s=200,
            depthshade=False,
            zorder=10,
        )
        label = f"{stability} ({theta:.2f}, {r:.2f}, {rho:.2f}, {mean_val:.2f})"
        legend_handles.append(
            Line2D(
                [],
                [],
                label=label,
                marker=mk,
                markerfacecolor=clr,
                markeredgecolor="black",
                markersize=10,
                linestyle="",
            )
        )
    ax.legend(
        handles=legend_handles,
        title="stability (\u03b8, r, \u03c1, migration coherence)",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.05),
        fontsize=8,
    )

    ax.set_xlabel(x_col, labelpad=2)
    ax.set_ylabel(y_col, labelpad=2)
    ax.set_zlabel(z_col, rotation=90, labelpad=2)
    dataset_config = load_dataset_config(dataset_name)
    shear_stress_values = [fc.shear_stress for fc in dataset_config.flow_conditions]
    shear_stress_label = "-".join(f"{v:g}" for v in shear_stress_values)
    ax.set_title(f"{dataset_name}, {shear_stress_label} dyn/cm²", loc="left")
    fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.6)
    fig.subplots_adjust(left=0.05, right=0.95)
    plt.show()
    save_plot_to_path(
        fig,
        output_dir,
        f"{dataset_name}_3D_{'binned' if binned else 'scatter'}_{color_col}",
    )
    plt.close(fig)
    return fig, ax


def plot_fixed_points_vs_shear_stress(
    df_fp: pd.DataFrame,
    variable: str,
    label: str,
    output_dir: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot a single fixed-point variable vs shear stress.

    Parameters
    ----------
    df_fp
        Concatenated fixed-points dataframe with a ``"shear_stress"`` column
        (e.g. from :func:`add_shear_stress_to_df`).
    variable
        Column name to plot on the y-axis.
    label
        Display label for the y-axis.
    output_dir
        Directory where the figure is saved.
    ylim
        Optional ``(ymin, ymax)`` limits for the y-axis.
    """
    from matplotlib.lines import Line2D

    # Sort shear stress categories from lowest to highest
    shear_order = sorted(
        df_fp["shear_stress"].unique(),
        key=lambda s: float(s.split("-")[0]),
    )
    df_fp["shear_stress"] = pd.Categorical(
        df_fp["shear_stress"],
        categories=shear_order,
        ordered=True,
    )
    df_fp = df_fp.sort_values("shear_stress")

    # Build legend handles
    legend_handles = []
    for stability in df_fp[STABILITY_COLUMN_NAME].unique():
        mk = STABILITY_MARKER_DICT.get(stability, "o")
        clr = STABILITY_COLOR_DICT.get(stability, "gray")
        legend_handles.append(
            Line2D(
                [],
                [],
                marker=mk,
                markerfacecolor=clr,
                markeredgecolor="black",
                markersize=8,
                linestyle="",
                label=stability,
            )
        )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for _, row in df_fp.iterrows():
        stability = row[STABILITY_COLUMN_NAME]
        mk = STABILITY_MARKER_DICT.get(stability, "o")
        clr = STABILITY_COLOR_DICT.get(stability, "gray")
        ax.scatter(
            row["shear_stress"],
            row[variable],
            marker=mk,
            color=clr,
            edgecolor="black",
            linewidths=0.8,
            s=80,
            zorder=3,
        )
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_ylabel(label, fontsize=10)
    ax.set_xlabel("shear stress (dyn/cm\u00b2)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        title="stability",
    )
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    save_plot_to_path(fig, output_dir, f"fixed_points_{variable}_vs_shear_stress")
    plt.close(fig)
