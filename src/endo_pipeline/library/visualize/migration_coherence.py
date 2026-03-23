import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_2d, binned_statistic_dd

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import check_required_columns_in_dataframe
from endo_pipeline.library.visualize.diffae_features.pplane import make_legend_handles_for_fixed_pts
from endo_pipeline.settings.flow_field_dataframes import (
    STABILITY_COLOR_DICT,
    STABILITY_COLUMN_NAME,
    STABILITY_MARKER_DICT,
    StabilityLegendHandle,
)
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


def plot_3d_scatter_or_binned(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str,
    df_fp: pd.DataFrame | None = None,
    binned: bool = False,
    bin_size_xyz: tuple[float, float, float] = (MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,) * 3,
    cmap: str = MIGRATION_COHERENCE_COLORMAP,
    vmin: float = 0,
    vmax: float = 1,
) -> tuple[plt.Figure, Axes3D]:
    """
    Plot a 3D scatter or 3D binned heatmap with optional fixed-point overlay.

    Parameters
    ----------
    df
        Dataframe containing columns *x_col*, *y_col*, *z_col*, and *color_col*.
    x_col, y_col, z_col
        Column names for the three spatial axes.
    color_col
        Column name whose values are mapped to color.
    df_fp
        Fixed-points dataframe. If provided, fixed points are overlaid with
        stability-specific markers and colors. If ``None``, no overlay is drawn.
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

    ax: Axes3D
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
    ax.computed_zorder = False

    if not binned:
        sc = ax.scatter(
            xs=x,
            ys=y,
            zs=z,
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
            xs=gx.ravel()[mask],
            ys=gy.ravel()[mask],
            zs=gz.ravel()[mask],
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
    legend_handles = []
    if df_fp is not None:
        for _, row in df_fp.iterrows():
            stability = row[STABILITY_COLUMN_NAME]
            mk = STABILITY_MARKER_DICT.get(stability, "o")
            clr = STABILITY_COLOR_DICT.get(stability, "gray")
            theta, r, rho = row[x_col], row[y_col], row[z_col]
            mean_val = row.get(f"mean_{color_col}", float("nan"))
            ax.scatter(
                xs=[theta],
                ys=[r],
                zs=[rho],
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
                StabilityLegendHandle(
                    stability_label=stability,
                    legend_label=label,
                )
            )
    if legend_handles:
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

    fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.6)
    fig.subplots_adjust(left=0.05, right=0.95)
    return fig, ax


def plot_fixed_points_vs_shear_stress(
    df_fp: pd.DataFrame,
    variable: str,
    label: str,
    output_dir: Path,
    ylim: tuple[float, float] | None = None,
    marker_size_scatter: int = 80,
    marker_size_legend: int = 8,
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
    legend_handles = make_legend_handles_for_fixed_pts(
        fpt_stabilities=df_fp[STABILITY_COLUMN_NAME].unique().tolist(),
        marker_size=marker_size_legend,
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
            s=marker_size_scatter,
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
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    plt.show()
    save_plot_to_path(fig, output_dir, f"fixed_points_{variable}_vs_shear_stress")
    plt.close(fig)
