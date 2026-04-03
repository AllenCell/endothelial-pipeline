import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_2d, binned_statistic_dd

from endo_pipeline.configs import TimepointAnnotation, load_dataset_config
from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    check_required_columns_in_dataframe,
    filter_dataframe_by_annotations,
    split_dataset_by_flow,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_binned_mean_to_fixed_points,
    add_optical_flow_features,
    add_shear_stress_to_df,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
from endo_pipeline.library.visualize.diffae_features.pplane import make_legend_handles_for_fixed_pts
from endo_pipeline.manifests import DataframeManifest, get_dataframe_location_for_dataset
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.flow_field_dataframes import (
    STABILITY_COLOR_DICT,
    STABILITY_COLUMN_NAME,
    STABILITY_MARKER_DICT,
    StabilityLegendHandle,
)
from endo_pipeline.settings.migration_coherence import (
    MIGRATION_COHERENCE_COLORMAP,
    MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
    MIGRATION_COHERENCE_HIST_FIGSIZE,
    MIGRATION_COHERENCE_HIST_PLOT_KDE,
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
    summary_stats: list[dict] | None = None,
    by_dataset: bool = True,
    marker_size_scatter: int = 80,
    marker_size_legend: int = 8,
) -> None:
    """Plot a single fixed-point variable vs shear stress, per dataset.

    Optionally overlays per-dataset mean \u00b1 std error bars when
    *summary_stats* is provided.

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
    summary_stats
        Optional list of dicts with keys ``"shear_stress"``, ``"mean"``,
        ``"std"``, ``"color"``, and ``"label"`` for each dataset/flow
        condition.  When provided, mean \u00b1 std error bars are overlaid.
    by_dataset
        If ``True`` (default), each dataset gets its own categorical x
        position, sorted by shear stress. Tick labels show
        ``"dataset_name (shear_stress)"``.
        If ``False``, x positions are the numeric shear-stress values and
        datasets with the same shear stress overlap.
    """

    # Convert shear stress to numeric values for sorting
    df_fp = df_fp.copy()
    df_fp["shear_stress_numeric"] = df_fp["shear_stress"].apply(
        lambda s: round(float(s.split("-")[0])) if isinstance(s, str) else round(float(s))
    )
    df_fp = df_fp.sort_values("shear_stress_numeric")

    # Build legend handles for fixed-point stability markers
    legend_handles = make_legend_handles_for_fixed_pts(
        fpt_stabilities=df_fp[STABILITY_COLUMN_NAME].unique().tolist(),
        marker_size=marker_size_legend,
    )

    if by_dataset:
        # Categorical x-axis: one tick per dataset
        unique_datasets = df_fp["dataset"].unique()
        row_to_x = lambda row: {d: i for i, d in enumerate(unique_datasets)}[
            row["dataset"]
        ]  # noqa: E731
        tick_positions = list(range(len(unique_datasets)))
        tick_labels = [
            f"{d} ({df_fp.loc[df_fp['dataset'] == d, 'shear_stress_numeric'].iloc[0]})"
            for d in unique_datasets
        ]
        fig_width = len(unique_datasets) * 1.2
    else:
        # Numeric x-axis: position by shear stress value
        row_to_x = lambda row: row["shear_stress_numeric"]  # noqa: E731
        unique_shear = sorted(df_fp["shear_stress_numeric"].unique())
        tick_positions = unique_shear
        tick_labels = [str(int(s)) for s in unique_shear]
        fig_width = max(8, len(unique_shear) * 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, 3.5))

    # Overlay mean ± std error bars when summary stats are provided
    if summary_stats is not None:
        summary_sorted = sorted(summary_stats, key=lambda s: s["shear_stress"])
        for i, stat in enumerate(summary_sorted):
            x_pos = i if by_dataset else stat["shear_stress"]
            ax.errorbar(
                x_pos,
                stat["mean"],
                yerr=stat["std"],
                fmt="o",
                color=stat["color"],
                markersize=8,
                capsize=4,
                elinewidth=1.5,
                label=stat["label"],
                zorder=2,
            )

    # Plot fixed points — draw gray (unknown stability) behind others
    for _, row in df_fp.iterrows():
        stability = row[STABILITY_COLUMN_NAME]
        mk = STABILITY_MARKER_DICT.get(stability, "o")
        clr = STABILITY_COLOR_DICT.get(stability, "gray")
        is_gray = stability not in STABILITY_COLOR_DICT
        ax.scatter(
            row_to_x(row),
            row[variable],
            marker=mk,
            color=clr,
            edgecolor="black",
            linewidths=0.8,
            s=marker_size_scatter,
            alpha=0.35 if is_gray else 1.0,
            zorder=1 if is_gray else 3,
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
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


def plot_optical_flow_histogram(
    df: pd.DataFrame,
    optical_flow_feature: str,
    title: str,
    color: str,
    output_dir: Path,
    filename: str,
    df_fp: pd.DataFrame | None = None,
) -> None:
    """Plot and save a histogram of an optical flow feature for a single dataset/flow condition.

    Parameters
    ----------
    df
        DataFrame containing the optical flow feature column.
    optical_flow_feature
        Name of the optical flow feature column to plot.
    title
        Plot title (e.g. dataset name and shear stress).
    color
        Color of the histogram bars.
    output_dir
        Directory where the figure is saved.
    filename
        Filename (without extension) for the saved figure.
    df_fp
        Optional fixed-points dataframe. If provided, each fixed point's
        ``mean_{optical_flow_feature}`` value is overlaid as a marker on
        the x-axis, colored and shaped by its stability classification.
    """

    data = df[optical_flow_feature].dropna()
    mean = data.mean()
    median = data.median()
    std = data.std()
    cov = std / mean

    if "unit_vector" in optical_flow_feature:
        binwidth = 0.02
    if "speed" in optical_flow_feature:
        binwidth = 0.2

    fig, ax = plt.subplots(figsize=MIGRATION_COHERENCE_HIST_FIGSIZE)
    sns.histplot(
        data,
        kde=MIGRATION_COHERENCE_HIST_PLOT_KDE,
        binwidth=binwidth,
        ax=ax,
        color=color,
    )
    ax.axvline(mean, color="black", linestyle="--", linewidth=1)
    ax.axvline(median, color="grey", linestyle="-", linewidth=1)
    ax.text(
        0.05,
        0.95,
        f"N = {len(data)}\n\u03bc = {mean:.3f}\n\u03c3 = {std:.3f} \nCOV = {cov:.3f} \nmedian = {median:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75},
    )

    # Overlay fixed points on x-axis
    if df_fp is not None:
        mean_col = f"mean_{optical_flow_feature}"
        if mean_col in df_fp.columns:
            for _, row in df_fp.iterrows():
                stability = row[STABILITY_COLUMN_NAME]
                mk = STABILITY_MARKER_DICT.get(stability, "o")
                clr = STABILITY_COLOR_DICT.get(stability, "gray")
                fp_val = row[mean_col]
                if pd.notna(fp_val):
                    ax.scatter(
                        fp_val,
                        0,
                        marker=mk,
                        color=clr,
                        edgecolor="black",
                        linewidths=1,
                        s=120,
                        zorder=5,
                        clip_on=False,
                    )
        filename = f"{filename}_with_fixed_points" if df_fp is not None else filename

    ax.set_xlabel(optical_flow_feature)
    if optical_flow_feature == "optical_flow_mean_speed_dt1":
        ax.set_xlim(0, 10)
    if optical_flow_feature == "ema01_optical_flow_mean_unit_vector_dt1":
        ax.set_xlim(0, 1)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    save_plot_to_path(fig, output_dir, filename)
    plt.close(fig)


def plot_cross_dataset_summaries(
    dataset_names: list[str],
    optical_flow_feature: str,
    feature_dataframe_manifest: DataframeManifest,
    fixed_points_dataframe_manifest: DataframeManifest,
    output_dir: Path,
    plot_fixed_points: bool = True,
    by_dataset: bool = True,
) -> None:
    """Compute and plot cross-dataset summary visualizations.

    This function loads all necessary data from the provided manifests and
    produces:

    - Fixed-point variable vs shear-stress plots (polar angle, polar radius,
      PC3, and mean optical-flow feature).
    - A mean-coherence vs shear-stress summary plot.

    Parameters
    ----------
    dataset_names
        List of dataset names to include in the summaries.
    optical_flow_feature
        Name of the optical-flow feature column to summarise.
    feature_dataframe_manifest
        Manifest containing per-dataset feature dataframe locations.
    fixed_points_dataframe_manifest
        Manifest containing per-dataset fixed-point dataframe locations.
    output_dir
        Directory where the figures are saved.
    plot_fixed_points
        Whether to compute and overlay fixed-point data on the summaries.
    by_dataset
        If ``True`` (default), each dataset gets its own categorical x position in the fixed
        point vs shear stress plot, with tick labels showing dataset name and shear stress.
        If ``False``, x positions are the numeric shear-stress values and datasets with the
        same shear stress overlap.
    """
    summary_stats: list[dict[str, float | str]] = []
    df_fp_all_list: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found for dataset [ %s ]. Skipping.",
                dataset_name,
            )
            continue

        # Load, filter, and enrich the feature dataframe
        df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]
        df_ = df[columns_to_compute].compute()
        df_steady_state = filter_dataframe_by_annotations(
            df_,
            load_dataset_config(dataset_name),
            timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
        )
        df_of = add_optical_flow_features(df_steady_state, datasets=[dataset_name])

        dataset_config = load_dataset_config(dataset_name)
        df_by_flow, shear_stress_list = split_dataset_by_flow(df_of, dataset_config)
        hist_color = get_dataset_color(dataset_name)

        for df_flow, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            plot_label = f"{dataset_name} ({int(shear_stress)} dyn/cm$^2$)"

            # Summary stats for the mean-coherence plot
            flow_mean = df_flow[optical_flow_feature].mean()
            flow_std = df_flow[optical_flow_feature].std()
            flow_cov = flow_std / flow_mean if flow_mean != 0 else float("nan")
            summary_stats.append(
                {
                    "label": plot_label,
                    "shear_stress": int(shear_stress),
                    "mean": flow_mean,
                    "std": flow_std,
                    "cov": flow_cov,
                    "color": hist_color,
                }
            )

            # Fixed points with binned means
            if plot_fixed_points:
                try:
                    fp_location = get_dataframe_location_for_dataset(
                        fixed_points_dataframe_manifest, dataset_name
                    )
                    fp_df = load_dataframe(fp_location, delay=False)
                    check_required_columns_in_dataframe(
                        fp_df,
                        required_columns=[
                            *DYNAMICS_COLUMN_NAMES,
                            ColumnName.DATASET,
                            STABILITY_COLUMN_NAME,
                        ],
                    )
                    df_flow_no_nan = df_flow.dropna(subset=[optical_flow_feature])
                    fp_df = add_binned_mean_to_fixed_points(
                        fp_df,
                        df_flow_no_nan,
                        x_col=ColumnName.DiffAEData.POLAR_ANGLE,
                        y_col=ColumnName.DiffAEData.POLAR_RADIUS,
                        z_col=ColumnName.DiffAEData.PC3_FLIPPED,
                        binned_col=optical_flow_feature,
                    )
                    df_fp_all_list.append(fp_df)
                except KeyError:
                    logger.warning(
                        "No fixed point dataframe found for dataset [ %s ]. "
                        "Skipping fixed points.",
                        dataset_name,
                    )

    # --- Fixed-points vs shear stress ---
    if df_fp_all_list:
        df_fp_all = pd.concat(df_fp_all_list, ignore_index=True)
        df_fp_all = add_shear_stress_to_df(df_fp_all)

        mean_of_col = f"mean_{optical_flow_feature}"
        variables = [
            ColumnName.DiffAEData.POLAR_ANGLE,
            ColumnName.DiffAEData.POLAR_RADIUS,
            ColumnName.DiffAEData.PC3_FLIPPED,
            mean_of_col,
        ]
        labels = ["\u03b8", "r", "\u03c1", f"mean_{optical_flow_feature}"]

        for var, label in zip(variables, labels, strict=False):
            # For the mean-optical-flow variable, overlay per-dataset mean ± std
            # stats = summary_stats if var == mean_of_col and summary_stats else None
            plot_fixed_points_vs_shear_stress(
                df_fp_all,
                var,
                label,
                output_dir=output_dir,
                ylim=None,
                summary_stats=None,
                by_dataset=by_dataset,
            )

    # --- COV vs shear stress ---
    x = [s["shear_stress"] for s in summary_stats]
    y = [s["cov"] for s in summary_stats]
    colors = [s["color"] for s in summary_stats]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color=colors, s=80)
    ax.set_xlabel("shear stress (dyn/cm$^2$)")
    ax.set_ylabel("COV of migration coherence")
    plt.tight_layout()
    plt.show()
    save_plot_to_path(fig, output_dir, "migration_coherence_cov_vs_shear_stress")
