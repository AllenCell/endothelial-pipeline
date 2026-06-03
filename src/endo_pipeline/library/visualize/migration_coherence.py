"""Methods for visualizing migration coherence metrics and their relationships to morphology dynamics."""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_2d, binned_statistic_dd

from endo_pipeline.configs.dataset_config import TimepointAnnotation
from endo_pipeline.configs.dataset_config_io import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_annotations,
    filter_dataframe_to_flow_condition_by_timepoint,
)
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.fixed_points import StabilityLegendHandle
from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL, FONTSIZE_XSMALL
from endo_pipeline.settings.migration_coherence import (
    MIGRATION_COHERENCE_COLORMAP,
    MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
)
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME

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
    """Plot scatter plot and binned mean heatmap in 2D colored by a specified feature column.

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
    figsize
        Figure size for the overall plot (width, height).
    scatter_point_size
        Size of the points in the scatter plot.

    Returns
    -------
    :
        The created matplotlib Figure object.
    :
        Array of Axes objects corresponding to the scatter and heatmap panels.

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
    axs[0].text(
        0.05,
        0.95,
        f"N = {len(x)}",
        transform=axs[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

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
    vmin: float = 0.3,
    vmax: float = 1,
    figsize: tuple[float, float] = (8, 8),
) -> tuple[plt.Figure, Axes3D]:
    """Plot a 3D scatter or 3D binned heatmap with optional fixed-point overlay.

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

    Returns
    -------
    :
        The created matplotlib Figure object.
    :
        The Axes3D object containing the plot.

    """
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    z = df[z_col].to_numpy()
    c = df[color_col].to_numpy()

    x_bin_size, y_bin_size, z_bin_size = bin_size_xyz

    ax: Axes3D
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=1, ncols=3, width_ratios=[15, 3, 1], left=0.0, right=0.8, wspace=0.1
    )
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    cax = fig.add_subplot(gs[0, 2])
    ax.computed_zorder = False

    if not binned:
        sc = ax.scatter(
            xs=x,
            ys=y,
            zs=z,
            c=c,
            cmap=cmap,
            s=2,
            lw=0,
            marker=".",
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,
            zorder=1,
        )
        cbar_label = get_label_for_column(color_col).replace("\n", " ")
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
            stability = row[Column.VectorField.STABILITY]
            mk = FIXED_POINT_PLOT_STYLE[stability].marker
            clr = FIXED_POINT_PLOT_STYLE[stability].color
            theta, r, rho = row[x_col], row[y_col], row[z_col]
            ax.scatter(
                xs=[theta],
                ys=[r],
                zs=[rho],
                marker=mk,
                color=clr,
                edgecolor="black",
                linewidths=0.5,
                s=10,
                depthshade=False,
                zorder=10,
            )
            label = f"{stability} fixed point ({theta:.2f}, {r:.2f}, {rho:.2f})"
            legend_handles.append(
                StabilityLegendHandle(
                    stability_label=stability,
                    legend_label=label,
                    marker_size=5,
                )
            )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title=f"stability ({Unicode.THETA}, r, {Unicode.RHO})",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.15),
            fontsize=FONTSIZE_SMALL,
        )

    ax.set_xlabel(get_label_for_column(x_col), labelpad=2)
    ax.set_ylabel(get_label_for_column(y_col), labelpad=2)
    ax.set_zlabel(get_label_for_column(z_col), rotation=0, labelpad=0)

    # Apply ticks/tick_labels from column metadata when available
    for axis, col in [("x", x_col), ("y", y_col), ("z", z_col)]:
        if col in COLUMN_METADATA:
            meta = COLUMN_METADATA[col]
            if meta.ticks is not None:
                ticks = list(meta.ticks)
                getattr(ax, f"set_{axis}ticks")(ticks)
                if meta.tick_labels is not None:
                    getattr(ax, f"set_{axis}ticklabels")(meta.tick_labels, fontsize=FONTSIZE_XSMALL)

    cbar = fig.colorbar(sc, cax=cax, label=cbar_label)

    # Apply ticks from color_col metadata to the colorbar
    if color_col in COLUMN_METADATA:
        color_meta = COLUMN_METADATA[color_col]
        if color_meta.ticks is not None:
            cbar.set_ticks(list(color_meta.ticks))

    return fig, ax


def plot_optical_flow_histogram(
    df: pd.DataFrame,
    optical_flow_feature: str,
    feature_label: str,
    feature_lim: tuple[float, float] | None,
    ss_label: str,
    color: str,
    df_fp: pd.DataFrame | None = None,
    binwidth: float = 0.2,
    figure: tuple[plt.Figure, plt.Axes] | None = None,
    legend_loc: str | None = "upper left",
) -> plt.Figure:
    """Plot and save a histogram of an optical flow feature for a single dataset/flow condition.

    Parameters
    ----------
    df
        DataFrame containing the optical flow feature column.
    optical_flow_feature
        Name of the optical flow feature column to plot.
    feature_label
        Label for the optical flow feature (e.g. "Unit Vector Mean" or "Speed Mean").
    ss_label
        Label for the shear stress (e.g. "10 dyn/cm²").
    color
        Color of the histogram bars.
    feature_lim
        Tuple specifying the limits for the x-axis (min, max). If None, limits are determined automatically.
    df_fp
        Optional fixed-points dataframe. If provided, each fixed point's
        ``mean_{optical_flow_feature}`` value is overlaid as a marker on
        the x-axis, colored and shaped by its stability classification.
    figure
        Optional ``(fig, ax)`` tuple to plot onto. If ``None``, a new figure is created.

    """
    if figure is None:
        fig, ax = plt.subplots(figsize=(2, 2))
    else:
        fig, ax = figure
    data = df[optical_flow_feature].dropna()
    mean = data.mean()
    std = data.std()
    cov = std / mean

    sns.histplot(
        data,
        kde=False,
        binwidth=binwidth,
        ax=ax,
        color=color,
    )

    if legend_loc is not None:
        stats_text = f"{ss_label}\nn={len(data)}\n{Unicode.MU}={mean:.2f}\nCV={cov:.2f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            color=color,
            fontsize=FONTSIZE_XSMALL,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
    elif legend_loc is None:
        bars: list[Rectangle] = [
            p
            for p in ax.patches
            if isinstance(p, Rectangle)
            and np.allclose(p.get_facecolor()[:3], plt.matplotlib.colors.to_rgb(color))
        ]
        if bars:
            peak_bar = max(bars, key=lambda b: b.get_height())
            peak_x = peak_bar.get_x()
            stats_text = f"n={len(data)}\n{Unicode.MU}={mean:.2f}\nCV={cov:.2f}"
            x_loc = peak_x - 0.3  # shift left from the peak
            if x_loc < ax.get_xlim()[0]:
                x_loc = (
                    ax.get_xlim()[0] + 0.02
                )  # if shifted label goes beyond left limit, shift it back inside
            # Bold label for shear stress
            ax.text(
                x_loc,
                0.98,
                ss_label,
                color=color,
                fontweight="bold",
                fontsize=FONTSIZE_XSMALL,
                ha="left",
                va="top",
                transform=ax.get_xaxis_transform(),
            )
            # Normal-weight stats below
            ax.text(
                x_loc,
                0.91,
                stats_text,
                color=color,
                fontsize=FONTSIZE_XSMALL,
                ha="left",
                va="top",
                transform=ax.get_xaxis_transform(),
            )

    # Overlay fixed points on x-axis
    if df_fp is not None:
        mean_col = f"mean_{optical_flow_feature}"
        if mean_col in df_fp.columns:
            for _, row in df_fp.iterrows():
                stability = row[Column.VectorField.STABILITY]
                mk = FIXED_POINT_PLOT_STYLE[stability].marker
                clr = FIXED_POINT_PLOT_STYLE[stability].color
                fp_val = row[mean_col]
                if pd.notna(fp_val):
                    ax.scatter(
                        fp_val,
                        0,
                        marker=mk,
                        color=clr,
                        edgecolor="black",
                        linewidths=1,
                        s=15,
                        zorder=5,
                        clip_on=False,
                    )

    ax.set_xlabel(feature_label, fontsize=FONTSIZE_MEDIUM)
    ax.set_ylabel("Count", fontsize=FONTSIZE_MEDIUM)
    if feature_lim is not None:
        ax.set_xlim(feature_lim)

    # reduce label and tick label padding
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)

    return fig


def build_box_for_3d_plot(
    bin_edges: tuple[tuple[float, float], ...],
) -> tuple:
    """Build the 8 corner vertices and 12 edges of a bin defined by the provided bin edges.

    Constructs a rectangular cuboid in feature space from the provided bin
    edges, then identifies the 12 axis-aligned edges by keeping only vertex
    pairs whose distance equals one of the three bin side lengths.

    Parameters
    ----------
    bin_edges
        A tuple of three ``(min, max)`` pairs defining the extent of the bin
        along each of the three feature axes. Expected to be a tuple of the form:
        ```
        (
            (x_min, x_max),
            (y_min, y_max),
            (z_min, z_max),
        )
        ```

    Returns
    -------
    vertices : list[tuple[float, float, float]]
        The 8 corner vertices of the cuboid.
    edges : list[tuple[tuple, tuple]]
        The 12 axis-aligned edges, each represented as a pair of vertices.
    """

    bin_sizes = np.absolute(np.subtract.reduce(bin_edges, axis=1))

    xs, ys, zs = np.meshgrid(*bin_edges)
    xs = xs.ravel()
    ys = ys.ravel()
    zs = zs.ravel()

    vertices = list(zip(xs, ys, zs, strict=True))
    edges = []
    for v1, v2 in combinations(vertices, r=2):
        edge_length = np.linalg.norm(np.array(v1) - np.array(v2))
        if np.isclose(edge_length, bin_sizes).any():
            edges.append((v1, v2))
    return vertices, edges


def make_example_migration_coherence(
    dataset_name: str,
    figure_size: tuple[float, float],
    output_dir: Path,
    fig_name: str | None = None,
) -> None:

    dataset_config = load_dataset_config(dataset_name)

    feature_column_names = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *feature_column_names]

    optical_flow_feature = Column.OpticalFlow.UNIT_VECTOR_MEAN
    of_metadata = COLUMN_METADATA[optical_flow_feature]
    vmin = of_metadata.min
    vmax = of_metadata.max
    fig_name = fig_name or f"{dataset_name}_3D_scatter_{optical_flow_feature}"

    # load dataframe and perform additional filtering (remove
    # non-steady-state timepoints based on annotations), computing
    # only the columns needed for visualization/analysis
    feature_dataframe_manifest_name = GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)
    df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)

    df_ = df[columns_to_compute].compute()
    df_steady_state = filter_dataframe_by_annotations(
        df_,
        dataset_config,
        timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
    )

    df_of = add_optical_flow_features(
        df_steady_state,
        datasets=[dataset_name],
    )

    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)
    for flow_condition in dataset_config.flow_conditions:

        x_col_name, y_col_name, z_col_name = feature_column_names

        half_bin_size = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE / 2
        bin_centers = tuple(
            float(fixed_points_df[col].item()) for col in (x_col_name, y_col_name, z_col_name)
        )
        bin_edges: tuple[tuple[float, float], ...] = tuple(
            (center - half_bin_size, center + half_bin_size) for center in bin_centers
        )
        _, edges = build_box_for_3d_plot(bin_edges=bin_edges)

        df_flow = filter_dataframe_to_flow_condition_by_timepoint(
            df_of, dataset_config, flow_condition
        )
        df_flow_no_nan = df_flow.dropna(subset=[optical_flow_feature])

        # 3D Scatter
        fig, ax = plot_3d_scatter_or_binned(
            df_flow_no_nan,
            x_col=x_col_name,
            y_col=y_col_name,
            z_col=z_col_name,
            color_col=optical_flow_feature,
            df_fp=fixed_points_df,
            binned=False,
            vmax=vmax,
            vmin=vmin,
            figsize=figure_size,
        )
        # draw cube around bin edges
        for e_xyz in edges:
            ax.plot(*list(zip(*e_xyz, strict=True)), ls="-", lw=1, c="black", alpha=0.6)

        save_plot_to_path(fig, output_dir, fig_name, file_format=".svg", transparent=True)
        plt.close(fig)
