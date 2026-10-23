"""Plotting helpers for the analysis-window sweep of the fixed points."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    get_start_of_steady_state_for_position,
    load_dataset_config,
)
from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.window_sweep import SweptBoundary
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.figure_4 import wrap_theta_for_vector_field_vis
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    POLAR_ANGLE_PERIOD,
    TIME_STEP_IN_HOURS,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)

# Fixed-point coordinate columns, one per swept dynamics dimension.
FIXED_POINT_COLUMNS = {
    column_name: ColumnTemplate.FIXED_POINT % column_name for column_name in DYNAMICS_COLUMN_NAMES
}

STEADY_STATE_LINE_COLOR = "tab:green"
CELL_PILING_LINE_COLOR = "tab:red"
BASELINE_LINE_STYLE = "--"
BASELINE_LINE_WIDTH = 1.3
BASELINE_LINE_LABEL = "baseline (annotated window)"

BOUNDARY_ANCHOR = {
    SweptBoundary.START_STEADY_STATE: ("onset", min),
    SweptBoundary.START_CELL_PILING: ("cell_piling", max),
}
"""Anchor type and aggregation function for each swept boundary."""

RELATIVE_XLABEL = {
    SweptBoundary.START_STEADY_STATE: "Time from first steady-state start (hours)",
    SweptBoundary.START_CELL_PILING: "Time from last cell-piling start (hours)",
}
"""X-axis label for each swept boundary."""


def get_ylimits_for_column(column_name: str) -> tuple[float, float] | None:
    """Y-axis limits for a fixed-point coordinate, matching the paper figures.

    Theta (polar angle) uses the special vector-field range from the paper
    (``VECTOR_FIELD_THETA_RANGE``) rather than its raw metadata limits; the other
    coordinates fall back to their column-metadata limits.
    """
    if column_name == Column.DiffAEData.POLAR_ANGLE:
        return VECTOR_FIELD_THETA_RANGE
    y_min, y_max = COLUMN_METADATA[column_name].limits
    if isinstance(y_min, (int, float)) and isinstance(y_max, (int, float)):
        return (y_min, y_max)
    return None


def _wrap_theta_for_plot(theta_values: pd.Series) -> pd.Series:
    """Wrap polar-angle fixed points into the asymmetric display range defined
    by ``VECTOR_FIELD_THETA_RANGE``.

    Parameters
    ----------
    theta_values
        Series of polar-angle fixed points.

    Returns
    -------
    :
        Wrapped polar-angle values.
    """
    theta_min, theta_max = VECTOR_FIELD_THETA_RANGE
    folded = theta_values % POLAR_ANGLE_PERIOD
    return folded.transform(
        lambda theta, lo=theta_min, hi=theta_max: wrap_theta_for_vector_field_vis(theta, lo, hi)
    )


def get_annotation_hours(dataset_name: str) -> tuple[list[float], list[float]]:
    """Per-FOV steady-state onset and cell-piling start for a dataset, in hours.

    Times use the same fixed ``TIME_STEP_IN_MINUTES`` step as the sweep x-axis.
    Only FOVs with a real annotation contribute (positions without a cell-piling
    annotation are omitted rather than falling back to the movie end).
    """
    dataset_config = load_dataset_config(dataset_name)
    cell_piling_ranges = (dataset_config.timepoint_annotations or {}).get(
        TimepointAnnotation.CELL_PILING, {}
    )
    step_hours = TIME_STEP_IN_HOURS

    onset_hours: list[float] = []
    cell_piling_hours: list[float] = []
    for position in dataset_config.zarr_positions:
        onset = get_start_of_steady_state_for_position(dataset_config, position)
        if onset is not None:
            onset_hours.append(onset * step_hours)
        ranges = cell_piling_ranges.get(position, [])
        starts = [(r[0] if isinstance(r, (list, tuple)) else r) for r in ranges]
        if starts:
            cell_piling_hours.append(min(starts) * step_hours)
    return onset_hours, cell_piling_hours


def draw_annotation_lines(ax: plt.Axes, dataset_name: str) -> None:
    """Shade the min-max span of steady-state onset and cell-piling start across FOVs."""
    onset_hours, cell_piling_hours = get_annotation_hours(dataset_name)
    for values, color in (
        (onset_hours, STEADY_STATE_LINE_COLOR),
        (cell_piling_hours, CELL_PILING_LINE_COLOR),
    ):
        if not values:
            continue
        ax.axvspan(min(values), max(values), color=color, alpha=0.12, zorder=0)


def add_relative_time_axis(ax: plt.Axes, dataset_name: str, boundary: SweptBoundary) -> None:
    """Add a secondary top x-axis: time relative to the swept boundary's annotation.

    Annotations are per-FOV, so the axis is anchored to a single reference FOV:
    the first steady-state start (earliest onset) or the last cell-piling start
    (latest onset). This is a linear shift of the absolute-time axis.
    """
    onset_hours, cell_piling_hours = get_annotation_hours(dataset_name)
    which, reduce = BOUNDARY_ANCHOR.get(boundary, ("onset", min))
    values = onset_hours if which == "onset" else cell_piling_hours
    if not values:
        return
    anchor = float(reduce(values))

    def to_relative(x):
        return np.asarray(x, dtype=float) - anchor

    def to_absolute(x):
        return np.asarray(x, dtype=float) + anchor

    secondary = ax.secondary_xaxis("top", functions=(to_relative, to_absolute))
    secondary.set_xlabel(RELATIVE_XLABEL.get(boundary, "time from annotation (hours)"))


def draw_baseline_fixed_points(
    ax: plt.Axes,
    baseline_df: pd.DataFrame | None,
    fixed_point_column: str,
    column_name,
    shear_colors: dict | None = None,
    stable_only: bool = True,
) -> None:
    """Draw each baseline fixed point as a horizontal reference line on ``ax``.

    The baseline fixed points are computed with each FOV's own analysis window,
    so they do not depend on the swept edge position; a horizontal line (colored
    by shear-stress bin) marks the value across the whole sweep for comparison.
    """
    if baseline_df is None or baseline_df.empty:
        return
    if stable_only:
        baseline_df = baseline_df[
            baseline_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE
        ]
    for _, row in baseline_df.iterrows():
        y_value = row[fixed_point_column]
        if pd.isna(y_value):
            continue
        # Match the paper: wrap theta fixed points into the vector-field range.
        if column_name == Column.DiffAEData.POLAR_ANGLE:
            y_value = float(_wrap_theta_for_plot(pd.Series([y_value])).iloc[0])
        color = "black"
        if shear_colors is not None:
            color = shear_colors.get(row.get(Column.SHEAR_STRESS_BIN), "black")
        ax.axhline(
            y_value,
            color=color,
            linestyle=BASELINE_LINE_STYLE,
            linewidth=BASELINE_LINE_WIDTH,
            alpha=0.9,
            zorder=2,
        )


def annotation_legend_handles(dataset_name: str) -> list[Patch]:
    """Legend handles for the annotation shaded bands that have data."""
    onset_hours, cell_piling_hours = get_annotation_hours(dataset_name)
    handles = []
    if onset_hours:
        handles.append(
            Patch(
                facecolor=STEADY_STATE_LINE_COLOR,
                alpha=0.12,
                label="start of steady state (range)",
            )
        )
    if cell_piling_hours:
        handles.append(
            Patch(
                facecolor=CELL_PILING_LINE_COLOR,
                alpha=0.12,
                label="start of cell piling (range)",
            )
        )
    return handles


def plot_sweep(
    dataset_df: pd.DataFrame,
    dataset_name: str,
    boundary: SweptBoundary,
    x_max_hours: float,
    stable_only: bool = True,
    baseline_df: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot each fixed-point coordinate vs the swept edge's movie-time position.

    One subplot per swept coordinate; points colored by shear-stress bin and
    marked by stability. The x-axis is absolute movie time and shares the same
    range (``x_max_hours``) across both boundaries. Set ``stable_only`` to keep
    only stable fixed points. ``baseline_df`` (the fixed points computed with
    each FOV's own analysis window) is overlaid as horizontal reference lines.
    """
    if stable_only:
        dataset_df = dataset_df[dataset_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE]

    shear_bins = sorted(dataset_df[Column.SHEAR_STRESS_BIN].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(shear_bins), 1))
    shear_colors = {shear_bin: cmap(i) for i, shear_bin in enumerate(shear_bins)}

    fig, axes = plt.subplots(
        len(FIXED_POINT_COLUMNS),
        1,
        figsize=(5, 2.5 * len(FIXED_POINT_COLUMNS)),
        sharex=True,
        layout="constrained",
    )
    axes = axes if len(FIXED_POINT_COLUMNS) > 1 else [axes]

    for ax, (column_name, fixed_point_column) in zip(
        axes, FIXED_POINT_COLUMNS.items(), strict=False
    ):
        for shear_bin in shear_bins:
            shear_df = dataset_df[dataset_df[Column.SHEAR_STRESS_BIN] == shear_bin]
            for stability, style in FIXED_POINT_PLOT_STYLE.items():
                subset = shear_df[shear_df[Column.FIXED_POINT_STABILITY] == stability]
                if subset.empty:
                    continue
                # Match the paper: wrap theta fixed points into the vector-field range.
                y_values = subset[fixed_point_column]
                if column_name == Column.DiffAEData.POLAR_ANGLE:
                    y_values = _wrap_theta_for_plot(y_values)
                ax.scatter(
                    subset[Column.SteadyStateWindow.WINDOW_EDGE_HOURS],
                    y_values,
                    marker=style.marker,
                    s=style.markersize,
                    color=shear_colors[shear_bin],
                    edgecolors="black",
                    linewidths=0.4,
                    alpha=0.85,
                )
        ax.set_xlim(0, x_max_hours)
        # Fix the y-axis to each coordinate's limits (theta uses the paper
        # vector-field range, r and rho use their metadata limits) so panels
        # are comparable.
        ylimits = get_ylimits_for_column(column_name)
        if ylimits is not None:
            ax.set_ylim(*ylimits)
        # Use π-based tick labels for θ (matching the paper figures).
        metadata = COLUMN_METADATA[column_name]
        if column_name == Column.DiffAEData.POLAR_ANGLE and metadata.ticks is not None:
            ax.set_yticks(list(metadata.ticks))
            if metadata.tick_labels is not None:
                ax.set_yticklabels(metadata.tick_labels)
        # Mark the steady-state onset and cell-piling annotations.
        draw_annotation_lines(ax, dataset_name)
        # Overlay the baseline fixed points (per-FOV windows) as horizontal
        # reference lines so the swept estimates can be compared against them.
        draw_baseline_fixed_points(
            ax, baseline_df, fixed_point_column, column_name, shear_colors, stable_only
        )
        ax.set_ylabel(get_label_for_column(column_name))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Movie time (hours)")
    # Secondary top axis: time relative to the swept boundary's annotation.
    add_relative_time_axis(axes[0], dataset_name, boundary)

    # Two-part legend: color = shear-stress bin, marker = stability.
    shear_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=shear_colors[shear_bin],
            label=f"{int(shear_bin)} dyn/cm², {dataset_name}",
        )
        for shear_bin in shear_bins
    ]
    stability_handles = [
        Line2D(
            [],
            [],
            marker=style.marker,
            linestyle="",
            markerfacecolor="lightgrey",
            markeredgecolor="black",
            label=str(stability),
        )
        for stability, style in FIXED_POINT_PLOT_STYLE.items()
        if (dataset_df[Column.FIXED_POINT_STABILITY] == stability).any()
    ]
    baseline_handles = []
    if baseline_df is not None and not baseline_df.empty:
        baseline_handles.append(
            Line2D(
                [],
                [],
                color="black",
                linestyle=BASELINE_LINE_STYLE,
                label=BASELINE_LINE_LABEL,
            )
        )
    handles = (
        shear_handles
        + stability_handles
        + baseline_handles
        + annotation_legend_handles(dataset_name)
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 2),
        fontsize="small",
        frameon=False,
    )
    return fig


def plot_num_timepoints(
    dataset_df: pd.DataFrame,
    dataset_name: str,
    boundary: SweptBoundary,
    x_max_hours: float,
    stable_only: bool = False,
) -> plt.Figure:
    """Plot the number of window timepoints used at each swept edge position.

    One point per (edge position, shear-stress bin); colored by shear-stress
    bin. The count is the total number of timepoints pooled across all FOVs
    (positions) that went into the estimate. The x-axis is absolute movie time.
    Set ``stable_only`` to restrict to stable fixed points (so the counts match
    a stable-only sweep plot).
    """
    if stable_only:
        dataset_df = dataset_df[dataset_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE]

    shear_bins = sorted(dataset_df[Column.SHEAR_STRESS_BIN].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(shear_bins), 1))
    shear_colors = {shear_bin: cmap(i) for i, shear_bin in enumerate(shear_bins)}

    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")

    for shear_bin in shear_bins:
        # One row per (edge position, shear bin): the count is identical across
        # the fixed points found for that condition, so drop duplicates.
        counts = (
            dataset_df[dataset_df[Column.SHEAR_STRESS_BIN] == shear_bin]
            .drop_duplicates(subset=Column.SteadyStateWindow.WINDOW_EDGE_TIMEPOINT)
            .sort_values(Column.SteadyStateWindow.WINDOW_EDGE_HOURS)
        )
        ax.plot(
            counts[Column.SteadyStateWindow.WINDOW_EDGE_HOURS],
            counts[Column.SteadyStateWindow.NUM_WINDOW_TIMEPOINTS],
            marker="o",
            color=shear_colors[shear_bin],
            label=f"{int(shear_bin)} dyn/cm²",
        )

    ax.set_xlim(0, x_max_hours)
    draw_annotation_lines(ax, dataset_name)
    ax.set_xlabel("Movie time (hours)")
    ax.set_ylabel("Number of timepoints in window\n(all FOVs)")
    add_relative_time_axis(ax, dataset_name, boundary)
    ax.grid(True, alpha=0.3)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=shear_colors[b],
            label=f"{int(b)} dyn/cm², {dataset_name}",
        )
        for b in shear_bins
    ] + annotation_legend_handles(dataset_name)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 4),
        fontsize="small",
        frameon=False,
    )
    return fig


def get_middle_boundary_time(
    dataset_config: DatasetConfig, boundary: SweptBoundary, sweep_df: pd.DataFrame
) -> float:
    """
    Find the sweep boundary time point closest to the median of the annotated
    times for the given boundary.

    For example, if the boundary is the start of the steady state, this function
    finds the median of the annotated start times. Then, it finds the sweep
    boundary time point (units hours) closest to that median and returns it.

    Parameters
    ----------
    dataset_config
        The configuration for the dataset, including annotated timepoints.
    boundary
        The "steady state"boundary for which to find the midpoint.
    sweep_df
        The dataframe containing the sweep boundary time points.

    Returns
    -------
    float
        The sweep boundary time point closest to the median of the annotated times for the given boundary.

    """
    if boundary not in list(SweptBoundary):
        raise ValueError(f"Unsupported boundary type: {boundary}")
    sweep_times = sweep_df[Column.SteadyStateWindow.WINDOW_EDGE_HOURS].to_numpy()
    if boundary == SweptBoundary.START_STEADY_STATE:
        start_times: list[int] = []
        for position in dataset_config.zarr_positions:
            start_time = get_start_of_steady_state_for_position(dataset_config, position)
            if start_time is not None:
                start_times.append(start_time)
        median_annotation_time = float(np.median(start_times)) if start_times else 0.0
    elif boundary == SweptBoundary.START_CELL_PILING:
        end_times: list[int] = []
        for position in dataset_config.zarr_positions:
            piling_annotations = get_annotated_timepoints_for_position(
                dataset_config, position, annotations=[TimepointAnnotation.CELL_PILING]
            )
            start_piling = piling_annotations[0] if piling_annotations else None
            if start_piling is not None:
                end_times.append(start_piling)
        median_annotation_time = (
            float(np.median(end_times)) if end_times else dataset_config.duration
        )
    median_time_hrs = median_annotation_time * TIME_STEP_IN_HOURS
    closest_sweep_time = sweep_times[np.argmin(np.abs(sweep_times - median_time_hrs))]
    return closest_sweep_time


def plot_sweep_subset(
    sweep_dfs: dict[SweptBoundary, pd.DataFrame],
    baseline_df: pd.DataFrame,
    dataset_config: DatasetConfig,
    output_path: Path,
) -> Path:
    """
    Plot the subsets of the window sweeps plus or minus three hours around the
    midpoint of the boundary annotations for that dataset.

    For example, if boundary is "start_steady_state", this method plots the
    results of the sweep for steady state start ranging from three hours before
    to three hours after the midpoint of the steady-state start annotations for
    the given dataset.
    """

    fig, axes = plt.subplots(
        len(FIXED_POINT_COLUMNS),
        2,
        figsize=(3.15, 0.75 * len(FIXED_POINT_COLUMNS)),
        layout="constrained",
    )
    axes = axes if len(FIXED_POINT_COLUMNS) > 1 else [axes]
    marker_style = FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE]

    for boundary, axes_col in [
        (SweptBoundary.START_STEADY_STATE, axes[:, 0]),
        (SweptBoundary.START_CELL_PILING, axes[:, 1]),
    ]:
        # Subset the sweep dataframe to +/- 3 hours around the sweep boundary
        # time closest to the median of the annotations for that boundary.
        sweep_df = sweep_dfs[boundary]
        middle_boundary_time = get_middle_boundary_time(dataset_config, boundary, sweep_df)
        logger.debug("Middle boundary time: %s", middle_boundary_time)
        sweep_df = sweep_df[
            (sweep_df[Column.SteadyStateWindow.WINDOW_EDGE_HOURS] >= middle_boundary_time - 3)
            & (sweep_df[Column.SteadyStateWindow.WINDOW_EDGE_HOURS] <= middle_boundary_time + 3)
        ]

        for ax, (column_name, fixed_point_column) in zip(
            axes_col, FIXED_POINT_COLUMNS.items(), strict=False
        ):
            # Match the paper: wrap theta fixed points into the vector-field range.
            y_values = sweep_df[fixed_point_column]
            if column_name == Column.DiffAEData.POLAR_ANGLE:
                y_values = _wrap_theta_for_plot(y_values)
            ax.scatter(
                sweep_df[Column.SteadyStateWindow.WINDOW_EDGE_HOURS],
                y_values,
                marker=marker_style.marker,
                s=2 * marker_style.markersize,
                color=marker_style.color,
                edgecolors="black",
                linewidths=0.4,
            )
            # Update x-axis ticks to be centered at middle boundary time, and
            # ticks of 1 hour +/- 3 hours around the middle boundary time.
            tick_range = np.arange(middle_boundary_time - 3, middle_boundary_time + 4)
            ax.set_xticks(tick_range)
            # only include x-axis tick labels for the last row of plots
            if ax != axes_col[-1]:
                ax.set_xticklabels([""] * len(tick_range))
            # only label every other tick for readability
            else:
                ax.set_xticklabels(
                    [str(tick) if i % 2 == 0 else "" for i, tick in enumerate(tick_range)]
                )
            ax.set_xlim(middle_boundary_time - 3.1, middle_boundary_time + 3.1)
            # add vertical dashed line at the middle boundary time
            ax.axvline(
                middle_boundary_time,
                color="dimgray",
                linestyle="--",
                linewidth=0.8,
                label="Median of actual annotations",
            )
            # Fix the y-axis to each coordinate's limits (theta uses the paper
            # vector-field range, r and rho use their metadata limits) so panels
            # are comparable.
            ylimits = get_ylimits_for_column(column_name)
            if ylimits is not None:
                ax.set_ylim(*ylimits)
            # Use π-based tick labels for θ (matching the paper figures).
            metadata = COLUMN_METADATA[column_name]
            if column_name == Column.DiffAEData.POLAR_ANGLE and metadata.ticks is not None:
                metadata_ticks = metadata.ticks
                if len(metadata_ticks) > 3:
                    metadata_ticks = metadata_ticks[::2]
                ax.set_yticks(metadata_ticks)
                if metadata.tick_labels is not None:
                    metadata_tick_labels = metadata.tick_labels
                    if len(metadata_tick_labels) > 3:
                        metadata_tick_labels = metadata_tick_labels[::2]
                    ax.set_yticklabels(metadata_tick_labels)
            # Overlay the baseline fixed points (per-FOV windows) as horizontal
            # reference lines so the swept estimates can be compared against them.
            draw_baseline_fixed_points(
                ax,
                baseline_df,
                fixed_point_column,
                column_name,
            )
            ax.grid(True, alpha=0.3)
            # Only add the y-axis label for the start of steady state boundary
            # (left-most plot in figure panel)
            if boundary == SweptBoundary.START_STEADY_STATE:
                column_label = get_label_for_column(column_name) or column_name
                labelpad = 11 if column_name == Column.DiffAEData.POLAR_ANGLE else 18
                ax.set_ylabel(
                    f"{column_label}$^*$", rotation=0, labelpad=labelpad, ha="left", va="center"
                )

        # x-axis label for the last row of plots
        if boundary == SweptBoundary.START_STEADY_STATE:
            xlabel = "Start of steady state\n(hrs, all FOVs)"
        elif boundary == SweptBoundary.START_CELL_PILING:
            xlabel = "End of steady state\n(hrs, all FOVs)"
        axes_col[-1].set_xlabel(xlabel)

    # Add suptitle with shear stress and replicate information
    shear_stress_bin = dataset_config.flow_conditions[-1].shear_stress_bin
    shear_stress_label = f"{shear_stress_bin} dyn/cm{Unicode.SQUARED}"
    replicate_number = dataset_config.replicate_number

    axes[0][0].set_title(f"{shear_stress_label} (replicate {replicate_number})", x=1.25)

    file_name = f"{dataset_config.name}_window_sweep"
    file_path = save_plot_to_path(
        fig, output_path, file_name, file_format=".svg", tight_layout=False
    )

    return file_path
