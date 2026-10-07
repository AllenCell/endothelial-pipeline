"""Plotting helpers for the analysis-window sweep of the fixed points."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_start_of_steady_state_for_position,
    load_dataset_config,
)
from endo_pipeline.library.analyze.window_sweep import (
    NUM_WINDOW_TIMEPOINTS,
    SHEAR_STRESS_BIN,
    WINDOW_EDGE_TIMEPOINT,
)
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.figure_4 import wrap_theta_for_vector_field_vis
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    POLAR_ANGLE_PERIOD,
    TIME_STEP_IN_MINUTES,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE

# Derived column added for plotting (swept edge position in movie-time hours).
WINDOW_EDGE_HOURS = "window_edge_hours"
MINUTES_PER_HOUR = 60

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
    "start_steady_state": ("onset", min),
    "start_cell_piling": ("cell_piling", max),
}
RELATIVE_XLABEL = {
    "start_steady_state": "Time from first steady-state start (hours)",
    "start_cell_piling": "Time from last cell-piling start (hours)",
}


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


def wrap_theta_for_paper(theta_values: pd.Series) -> pd.Series:
    """Wrap polar-angle fixed points into ``VECTOR_FIELD_THETA_RANGE`` (paper convention).

    Mirrors the figure-4 flow-field visualization: first fold into one period,
    then shift the tails into the asymmetric display range.
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
    step_hours = TIME_STEP_IN_MINUTES / 60

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


def add_relative_time_axis(ax: plt.Axes, dataset_name: str, boundary: str) -> None:
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
    shear_colors: dict,
    stable_only: bool,
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
            y_value = float(wrap_theta_for_paper(pd.Series([y_value])).iloc[0])
        color = shear_colors.get(row.get(SHEAR_STRESS_BIN), "black")
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
    boundary: str,
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

    shear_bins = sorted(dataset_df[SHEAR_STRESS_BIN].dropna().unique())
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
            shear_df = dataset_df[dataset_df[SHEAR_STRESS_BIN] == shear_bin]
            for stability, style in FIXED_POINT_PLOT_STYLE.items():
                subset = shear_df[shear_df[Column.FIXED_POINT_STABILITY] == stability]
                if subset.empty:
                    continue
                # Match the paper: wrap theta fixed points into the vector-field range.
                y_values = subset[fixed_point_column]
                if column_name == Column.DiffAEData.POLAR_ANGLE:
                    y_values = wrap_theta_for_paper(y_values)
                ax.scatter(
                    subset[WINDOW_EDGE_HOURS],
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
    boundary: str,
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

    shear_bins = sorted(dataset_df[SHEAR_STRESS_BIN].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(shear_bins), 1))
    shear_colors = {shear_bin: cmap(i) for i, shear_bin in enumerate(shear_bins)}

    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")

    for shear_bin in shear_bins:
        # One row per (edge position, shear bin): the count is identical across
        # the fixed points found for that condition, so drop duplicates.
        counts = (
            dataset_df[dataset_df[SHEAR_STRESS_BIN] == shear_bin]
            .drop_duplicates(subset=WINDOW_EDGE_TIMEPOINT)
            .sort_values(WINDOW_EDGE_HOURS)
        )
        ax.plot(
            counts[WINDOW_EDGE_HOURS],
            counts[NUM_WINDOW_TIMEPOINTS],
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
