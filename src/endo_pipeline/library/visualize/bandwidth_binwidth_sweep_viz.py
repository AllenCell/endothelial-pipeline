"""Methods for visualizing the kernel bandwidth and binwidth sweep results."""

import logging
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.bandwidth_binwidth_sweep import (
    get_param_sweep_run_name,
    get_parameter_space,
)
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings import plot_defaults
from endo_pipeline.settings.bandwidth_binwidth_sweep import (
    SWEEP_PARAM_EXAMPLE_BINWIDTHS,
    SWEEP_PARAM_EXAMPLE_KERNEL_BANDWIDTHS,
)
from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName, ColumnNameTemplate
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    DYNAMICS_COLUMN_NAMES,
    KERNEL_BANDWIDTHS_DYNAMICS,
    POLAR_ANGLE_PERIOD,
)
from endo_pipeline.settings.examples import SUPP_FIG_PARAM_SWEEP
from endo_pipeline.settings.figures import (
    FIGURE_SAVE_DPI,
    FONTSIZE_MEDIUM,
    FONTSIZE_SMALL,
    FONTSIZE_XSMALL,
    MAX_FIGURE_WIDTH,
)
from endo_pipeline.settings.manifest_names import GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME

logger = logging.getLogger(__name__)

# column names added to the sweep results to hold the sweep parameters of each run
BINWIDTH_COLUMN = "bw"
KERNEL_BANDWIDTH_COLUMN = "kb"
PARAM_COMBO_COLUMN = "param_combo"

# Shared horizontal layout (in inches) for the heatmap and fixed point location figures so that
# their dataset columns line up when the separately saved figures are stacked: the dataset columns
# of axes span PANEL_BLOCK_WIDTH, the colorbar of the heatmap figure is added in extra space to the
# right of that block. The block width is chosen so that the heatmap figure including its colorbar
# stays within MAX_FIGURE_WIDTH.
PANEL_BLOCK_WIDTH = 5.6
LEFT_MARGIN = 0.55
RIGHT_MARGIN = 0.05
WSPACE = 0.1
COLORBAR_PAD = 0.08
COLORBAR_WIDTH = 0.08
COLORBAR_LABEL_SPACE = 0.5

HEATMAP_FIG_HEIGHT = 1.9
HEATMAP_BOTTOM_MARGIN = 0.65
HEATMAP_TOP_MARGIN = 0.25
HEATMAP_FIG_WIDTH = PANEL_BLOCK_WIDTH + COLORBAR_PAD + COLORBAR_WIDTH + COLORBAR_LABEL_SPACE
assert HEATMAP_FIG_WIDTH <= MAX_FIGURE_WIDTH

# the bottom margin of the fixed point location figure is adjusted at draw time to fit the legend
# right underneath the x-axis labels, so the final figure height differs from FP_LOCATION_FIG_HEIGHT
# while the axes height (FP_LOCATION_FIG_HEIGHT - FP_LOCATION_BOTTOM_MARGIN - FP_LOCATION_TOP_MARGIN)
# is kept
FP_LOCATION_FIG_HEIGHT = 3.05
FP_LOCATION_BOTTOM_MARGIN = 1.05
FP_LOCATION_TOP_MARGIN = 0.25

FileFormat = Literal[".png", ".svg", ".pdf"]


def load_sweep_fixed_points(unwrap_angle: bool = True) -> pd.DataFrame:
    """
    Load the bootstrapped fixed points from every run of the kernel bandwidth and binwidth sweep
    into a single dataframe.

    Runs whose manifest has no locations (i.e. no bootstrapped fixed points were found) are skipped.

    Parameters
    ----------
    unwrap_angle
        True to unwrap the polar angle columns so that angles above 5pi/6 are shifted down by one
        period, avoiding fixed points being split across the periodic boundary in the plots.

    Returns
    -------
    :
        Dataframe with the bootstrapped fixed points of all datasets and all sweep runs, with the
        binwidth and kernel bandwidth of each run in the `BINWIDTH_COLUMN` and
        `KERNEL_BANDWIDTH_COLUMN` columns.
    """
    dataframes = []
    for bw, kb in get_parameter_space():
        sweep_run_name = get_param_sweep_run_name(bw, kb)
        manifest = load_dataframe_manifest(
            f"{GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME}_{sweep_run_name}"
        )
        if not manifest.locations:
            logger.info("Skipping parameters bw=%s, kb=%s due to empty manifest.", bw, kb)
            continue

        for dataset in manifest.locations:
            df = load_dataframe(manifest.locations[dataset])
            df[BINWIDTH_COLUMN] = bw
            df[KERNEL_BANDWIDTH_COLUMN] = kb
            dataframes.append(df)

    df_sweep = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

    baseline_angle_column = (
        ColumnNameTemplate.BASELINE_FIXED_POINT % ColumnName.DiffAEData.POLAR_ANGLE
    )
    if unwrap_angle and baseline_angle_column in df_sweep.columns:
        # unwrap baseline, bootstrapped cluster mean angle, and CI bounds
        for template in [
            ColumnNameTemplate.BASELINE_FIXED_POINT,
            ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN,
            ColumnNameTemplate.BOOTSTRAP_CI_LOWER,
            ColumnNameTemplate.BOOTSTRAP_CI_UPPER,
        ]:
            angle_column = template % ColumnName.DiffAEData.POLAR_ANGLE
            df_sweep[angle_column] = df_sweep[angle_column].apply(
                lambda angle: (angle - POLAR_ANGLE_PERIOD if angle > (5 * np.pi / 6) else angle)
            )

    return df_sweep


def _apply_shared_horizontal_layout(
    fig: Figure, fig_width: float, fig_height: float, bottom_margin: float, top_margin: float
) -> tuple[float, float]:
    """
    Position the dataset columns of axes of a figure according to the shared horizontal layout.

    Returns
    -------
    :
        Bottom and top of the axes as fractions of the figure height.
    """
    bottom = bottom_margin / fig_height
    top = 1 - top_margin / fig_height
    fig.subplots_adjust(
        left=LEFT_MARGIN / fig_width,
        right=(PANEL_BLOCK_WIDTH - RIGHT_MARGIN) / fig_width,
        bottom=bottom,
        top=top,
        wspace=WSPACE,
    )
    return bottom, top


def _count_fixed_points_on_sweep_grid(
    df_dataset_stability: pd.DataFrame,
    df_dataset: pd.DataFrame,
    binwidths: Sequence[float],
    kernel_bandwidths: Sequence[float],
) -> pd.DataFrame | None:
    """
    Count the fixed points of one dataset and stability that passed the bootstrap threshold for each
    parameter combination and lay the counts out on the full sweep grid.

    Parameter combinations that produced results for the dataset but had no fixed points of this
    stability pass the threshold are counted as 0, while combinations with no results at all are
    left as NaN.

    Parameters
    ----------
    df_dataset_stability
        Bootstrapped fixed points of one dataset and one stability.
    df_dataset
        Bootstrapped fixed points of the same dataset for all stabilities, used to determine which
        parameter combinations produced results.
    binwidths
        Binwidths forming the rows of the grid.
    kernel_bandwidths
        Kernel bandwidths forming the columns of the grid.

    Returns
    -------
    :
        Dataframe of counts with binwidths as index and kernel bandwidths as columns, or None if no
        fixed points passed the threshold.
    """
    df_passed = df_dataset_stability.loc[
        df_dataset_stability[ColumnName.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD
    ]
    if df_passed.empty:
        return None

    counts = (
        df_passed.groupby([KERNEL_BANDWIDTH_COLUMN, BINWIDTH_COLUMN], observed=False)
        .size()
        .reset_index(name="num_fixedpoints")
        .pivot(index=BINWIDTH_COLUMN, columns=KERNEL_BANDWIDTH_COLUMN, values="num_fixedpoints")
        .reindex(index=list(binwidths), columns=list(kernel_bandwidths))
    )

    successful_params = df_dataset[[BINWIDTH_COLUMN, KERNEL_BANDWIDTH_COLUMN]].drop_duplicates()
    for bw, kb in successful_params.itertuples(index=False):
        if bw in counts.index and kb in counts.columns and pd.isna(counts.at[bw, kb]):
            counts.at[bw, kb] = 0

    return counts


def plot_fixed_point_count_heatmaps(
    df_sweep: pd.DataFrame,
    stability: str,
    output_path: Path,
    datasets: Sequence[str] = SUPP_FIG_PARAM_SWEEP,
    file_format: FileFormat = ".png",
) -> Path:
    """
    Plot heatmaps of the number of fixed points of a given stability that passed the bootstrap
    threshold for every parameter combination of the sweep, one heatmap per dataset.

    The heatmaps share a single colorbar drawn to the right of the dataset columns. Parameter
    combinations with no results for a dataset are drawn in grey.

    Parameters
    ----------
    df_sweep
        Bootstrapped fixed points of all datasets and all sweep runs, as returned by
        `load_sweep_fixed_points`.
    stability
        Fixed point stability to plot (e.g. "stable" or "unstable").
    output_path
        Directory path to save the output figure.
    datasets
        Datasets to plot, in the order in which they appear as columns of the figure.
    file_format
        File format for the figure.

    Returns
    -------
    :
        Path to the saved figure file.
    """
    parameter_space = get_parameter_space()
    # seaborn draws the first row of a heatmap at the top, so order the binwidths from largest to
    # smallest to place the smallest binwidth at the bottom of the y-axis
    sweep_binwidths = sorted({bw for bw, _ in parameter_space}, reverse=True)
    sweep_kernel_bandwidths = sorted({kb for _, kb in parameter_space})

    df_datasets = df_sweep.loc[df_sweep[ColumnName.DATASET].isin(datasets)]
    df_stability = df_datasets.loc[df_datasets[ColumnName.FIXED_POINT_STABILITY] == stability]

    # use a common color range across all datasets so that the shared colorbar applies to every
    # heatmap
    max_num_fixedpoints = (
        df_stability.loc[df_stability[ColumnName.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD]
        .groupby([ColumnName.DATASET, KERNEL_BANDWIDTH_COLUMN, BINWIDTH_COLUMN], observed=True)
        .size()
        .max()
    )
    vmax = max(int(max_num_fixedpoints) if pd.notna(max_num_fixedpoints) else 0, 1)

    fig, axes = plt.subplots(
        figsize=(HEATMAP_FIG_WIDTH, HEATMAP_FIG_HEIGHT), ncols=len(datasets), sharey=True
    )
    bottom, top = _apply_shared_horizontal_layout(
        fig, HEATMAP_FIG_WIDTH, HEATMAP_FIG_HEIGHT, HEATMAP_BOTTOM_MARGIN, HEATMAP_TOP_MARGIN
    )

    for axis_index, (dataset, ax) in enumerate(zip(datasets, axes, strict=True)):
        heatmap_data = _count_fixed_points_on_sweep_grid(
            df_dataset_stability=df_stability.loc[df_stability[ColumnName.DATASET] == dataset],
            df_dataset=df_datasets.loc[df_datasets[ColumnName.DATASET] == dataset],
            binwidths=sweep_binwidths,
            kernel_bandwidths=sweep_kernel_bandwidths,
        )
        if heatmap_data is None:
            fig.delaxes(ax)
            continue

        # NaN cells are left unpainted by seaborn, letting the grey axes background show through
        ax.set_facecolor("lightgrey")
        sns.heatmap(
            heatmap_data,
            cmap="magma",
            linewidths=0.5,
            annot=True,
            annot_kws={"fontsize": FONTSIZE_XSMALL},
            ax=ax,
            vmin=0,
            vmax=vmax,
            cbar=False,
            # force a tick and label at every parameter value rather than letting seaborn thin them
            # out to avoid overlap
            xticklabels=[f"{kb:g}" for kb in heatmap_data.columns],
            yticklabels=[f"{bw:g}" for bw in heatmap_data.index],
        )

        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_XSMALL)
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)

        ax.set_title(dataset, fontsize=FONTSIZE_MEDIUM)
        ax.set_xlabel("Kernel Bandwidth", fontsize=FONTSIZE_SMALL)
        if axis_index == 0:
            ax.set_ylabel("Binwidth", fontsize=FONTSIZE_SMALL)
        else:
            ax.set(ylabel=None)

    # draw a single colorbar shared by all heatmaps in the extra space to the right of the dataset
    # columns, dropping the tickmarks at the boundaries between colors
    cbar_ax = fig.add_axes(
        (
            (PANEL_BLOCK_WIDTH + COLORBAR_PAD) / HEATMAP_FIG_WIDTH,
            bottom,
            COLORBAR_WIDTH / HEATMAP_FIG_WIDTH,
            top - bottom,
        )
    )
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap="magma"), cax=cbar_ax
    )
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=FONTSIZE_XSMALL)
    cbar.set_label(f"Number of {stability.capitalize()}\nFixed Points", fontsize=FONTSIZE_SMALL)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))

    # the layout is set explicitly above, so skip tight layout / bbox cropping that would shift
    # the axes
    return save_plot_to_path(
        figure=fig,
        output_path=output_path,
        figure_name=f"{stability}_heatmap",
        dpi=FIGURE_SAVE_DPI,
        file_format=file_format,
        pad_inches=0,
        tight_layout=False,
    )


def _format_param_combo(kernel_bandwidth: float, binwidth: float) -> str:
    """Format a (kernel bandwidth, binwidth) combination as a categorical axis label."""
    return f"({kernel_bandwidth}, {binwidth})"


def plot_fixed_point_location_summary(
    df_sweep: pd.DataFrame,
    stability: str,
    output_path: Path,
    datasets: Sequence[str] = SUPP_FIG_PARAM_SWEEP,
    file_format: FileFormat = ".png",
) -> Path:
    """
    Plot the locations of the fixed points of a given stability along each dynamics feature for
    the parameter combinations surrounding the ones used in the main analysis.

    The figure has one column per dataset and one row per dynamics feature, with a dashed line
    marking the parameter combination used in the main analysis. The columns use the same
    horizontal layout as `plot_fixed_point_count_heatmaps` so that the saved figures line up.

    Parameters
    ----------
    df_sweep
        Bootstrapped fixed points of all datasets and all sweep runs, as returned by
        `load_sweep_fixed_points`.
    stability
        Fixed point stability to plot (e.g. "stable" or "unstable").
    output_path
        Directory path to save the output figure.
    datasets
        Datasets to plot, in the order in which they appear as columns of the figure.
    file_format
        File format for the figure.

    Returns
    -------
    :
        Path to the saved figure file.
    """
    # the parameters that surround the one used in the main analysis (including the main analysis
    # parameters)
    local_parameter_space = set(
        product(SWEEP_PARAM_EXAMPLE_BINWIDTHS, SWEEP_PARAM_EXAMPLE_KERNEL_BANDWIDTHS)
    )
    bw_original = np.unique(list(BIN_WIDTHS_DYNAMICS.values())).item()
    kb_original = np.unique(list(KERNEL_BANDWIDTHS_DYNAMICS.values())).item()

    # rename the bootstrapped cluster mean columns to the plain feature names so that the column
    # metadata can be looked up directly
    cluster_mean_columns = {
        ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN % col: col for col in DYNAMICS_COLUMN_NAMES
    }
    df_plot = (
        df_sweep.loc[
            df_sweep[ColumnName.DATASET].isin(datasets)
            & (df_sweep[ColumnName.FIXED_POINT_STABILITY] == stability)
        ]
        .rename(columns=cluster_mean_columns)
        .copy()
    )
    df_plot = df_plot.loc[
        [
            (bw, kb) in local_parameter_space
            for bw, kb in zip(
                df_plot[BINWIDTH_COLUMN], df_plot[KERNEL_BANDWIDTH_COLUMN], strict=True
            )
        ]
    ]
    df_plot[PARAM_COMBO_COLUMN] = [
        _format_param_combo(kb, bw)
        for kb, bw in zip(df_plot[KERNEL_BANDWIDTH_COLUMN], df_plot[BINWIDTH_COLUMN], strict=True)
    ]
    df_plot = df_plot.sort_values(by=[KERNEL_BANDWIDTH_COLUMN, BINWIDTH_COLUMN])

    # share the y-axis within each row (one feature per row) and the x-axis within each column
    fig, axes = plt.subplots(
        figsize=(PANEL_BLOCK_WIDTH, FP_LOCATION_FIG_HEIGHT),
        ncols=len(datasets),
        nrows=len(DYNAMICS_COLUMN_NAMES),
        sharex=True,
        sharey="row",
    )
    _apply_shared_horizontal_layout(
        fig,
        PANEL_BLOCK_WIDTH,
        FP_LOCATION_FIG_HEIGHT,
        FP_LOCATION_BOTTOM_MARGIN,
        FP_LOCATION_TOP_MARGIN,
    )

    for axis_index, dataset in enumerate(datasets):
        df_dataset = df_plot.loc[df_plot[ColumnName.DATASET] == dataset]
        dataset_axes = axes[:, axis_index]

        # label each column of axes with the dataset name over the top row only
        dataset_axes[0].set_title(dataset, fontsize=FONTSIZE_MEDIUM)

        for col, ax in zip(DYNAMICS_COLUMN_NAMES, dataset_axes, strict=True):
            ax.scatter(
                x=df_dataset[PARAM_COMBO_COLUMN],
                y=df_dataset[col],
                alpha=0.5,
                c="black",
                marker=".",
            )
            column_metadata = COLUMN_METADATA[col]
            if col == ColumnName.DiffAEData.POLAR_ANGLE:
                ax.set_ylim(plot_defaults.SUMMARY_PLOT_THETA_RANGE)
            else:
                ax.set_ylim(column_metadata.min, column_metadata.max)
            if column_metadata.ticks is not None:
                ax.set_yticks(column_metadata.ticks)
            if column_metadata.tick_labels is not None:
                ax.set_yticklabels(column_metadata.tick_labels)

            # only show the y-axis label and ticklabels on the first column
            if axis_index == 0:
                ax.set_ylabel(f"{column_metadata.label_with_unit}$^*$", fontsize=FONTSIZE_SMALL)
                ax.tick_params(axis="y", labelsize=FONTSIZE_XSMALL)
            else:
                ax.set(ylabel=None)
                ax.tick_params(axis="y", labelleft=False)

            # mark the parameter combination used in the main analysis
            main_params_line = ax.axvline(
                x=_format_param_combo(kb_original, bw_original),
                color="red",
                linestyle="--",
                linewidth=1,
                zorder=0,
                label="Main analysis parameters",
            )

        dataset_axes[-1].set_xticklabels(
            dataset_axes[-1].get_xticklabels(), rotation=90, fontsize=FONTSIZE_XSMALL
        )
        dataset_axes[-1].set_xlabel("Parameters (kb, bw)", fontsize=FONTSIZE_SMALL)

    # add a single legend for the main analysis parameter line, centered under the dataset columns
    # right underneath the x-axis labels with a gap of half the legend font height. The x-axis
    # labels are only positioned at draw time, so draw the figure first to measure where they are
    fig.draw_without_rendering()
    xlabels_bottom = min(ax.xaxis.label.get_window_extent().y0 for ax in axes[-1]) / fig.dpi
    legend = fig.legend(
        handles=[main_params_line],
        loc="upper center",
        borderpad=0,
        borderaxespad=0,
        frameon=False,
        fontsize=FONTSIZE_XSMALL,
    )
    legend_height = legend.get_window_extent().height / fig.dpi
    legend_gap = 0.5 * FONTSIZE_XSMALL / 72

    # trim (or add) the space below the legend from the bottom margin and the figure height so the
    # legend sits at the bottom edge of the figure, keeping the size of the axes unchanged
    excess_height = xlabels_bottom - legend_gap - legend_height
    fig_height = FP_LOCATION_FIG_HEIGHT - excess_height
    fig.set_size_inches(PANEL_BLOCK_WIDTH, fig_height)
    _apply_shared_horizontal_layout(
        fig,
        PANEL_BLOCK_WIDTH,
        fig_height,
        FP_LOCATION_BOTTOM_MARGIN - excess_height,
        FP_LOCATION_TOP_MARGIN,
    )
    panel_block_center = (LEFT_MARGIN + PANEL_BLOCK_WIDTH - RIGHT_MARGIN) / 2 / PANEL_BLOCK_WIDTH
    legend.set_bbox_to_anchor(
        (panel_block_center, legend_height / fig_height), transform=fig.transFigure
    )

    # the layout is set explicitly above, so skip tight layout that would shift the axes
    return save_plot_to_path(
        figure=fig,
        output_path=output_path,
        figure_name=f"{stability}_fixed_point_positions",
        dpi=FIGURE_SAVE_DPI,
        file_format=file_format,
        pad_inches=0,
        tight_layout=False,
    )
