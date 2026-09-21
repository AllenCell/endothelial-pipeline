"""Classes and methods for visualizing fixed points."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE


class StabilityLegendHandle(Line2D):
    """Custom legend handle for fixed point stability classifications."""

    def __init__(
        self,
        stability_label: StabilityLabel,
        legend_label: str | None = None,
        marker: str | None = None,
        face_color: str | None = None,
        marker_size: int = 10,
        edge_color: str = "black",
    ):
        super().__init__(
            [],
            [],
            label=legend_label or stability_label.value,
            marker=marker or FIXED_POINT_PLOT_STYLE[stability_label].marker,
            color=face_color or FIXED_POINT_PLOT_STYLE[stability_label].color,
            markersize=marker_size,
            markeredgecolor=edge_color,
            linestyle="",
        )


def _get_asymmetric_error(value: float, lower: float, upper: float) -> list[list[float]] | None:
    """Build a matplotlib asymmetric error bar entry, or None if the CI is undefined."""
    if np.isnan(lower) or np.isnan(upper):
        return None
    return [[max(0.0, value - lower)], [max(0.0, upper - value)]]


def plot_bootstrap_fixed_points_on_axis(
    ax: plt.Axes,
    fixed_points_dataframe: pd.DataFrame,
    column_x: str,
    column_y: str,
    edge_color: str = "black",
    marker_size: int = 8,
    use_stability_style: bool = True,
    face_color: str | None = None,
) -> None:
    """Plot bootstrap fixed point cluster means with confidence interval error bars.

    Each fixed point is drawn at its bootstrap cluster mean with asymmetric error
    bars spanning the per-coordinate bootstrap confidence interval. Fixed points
    whose confidence interval is undefined (fewer than two bootstrap hits) are
    drawn without error bars.

    Parameters
    ----------
    ax
        Axis to plot on.
    fixed_points_dataframe
        Dataframe of bootstrap results, one row per baseline fixed point.
    column_x
        Feature column plotted on the x axis.
    column_y
        Feature column plotted on the y axis.
    edge_color
        Marker edge and error bar color, used to distinguish groups of results.
    marker_size
        Size of the fixed point markers.
    use_stability_style
        True to color and shape markers by stability classification, False to
        use ``face_color`` with a circular marker.
    face_color
        Marker face color used when ``use_stability_style`` is False.

    """
    for _, row in fixed_points_dataframe.iterrows():
        if use_stability_style:
            stability = row[Column.FIXED_POINT_STABILITY]
            color = FIXED_POINT_PLOT_STYLE[stability].color
            marker = FIXED_POINT_PLOT_STYLE[stability].marker
        else:
            color = face_color or edge_color
            marker = "o"

        x = row[ColumnTemplate.BOOTSTRAP_CLUSTER_MEAN % column_x]
        y = row[ColumnTemplate.BOOTSTRAP_CLUSTER_MEAN % column_y]

        ax.errorbar(
            x,
            y,
            xerr=_get_asymmetric_error(
                x,
                row[ColumnTemplate.BOOTSTRAP_CI_LOWER % column_x],
                row[ColumnTemplate.BOOTSTRAP_CI_UPPER % column_x],
            ),
            yerr=_get_asymmetric_error(
                y,
                row[ColumnTemplate.BOOTSTRAP_CI_LOWER % column_y],
                row[ColumnTemplate.BOOTSTRAP_CI_UPPER % column_y],
            ),
            fmt=marker,
            color=color,
            markeredgecolor=edge_color,
            markeredgewidth=1.5,
            markersize=marker_size,
            capsize=4,
            elinewidth=1.2,
            ecolor=edge_color,
            zorder=3,
        )


def get_stability_legend_handles(
    fixed_points_dataframes: list[pd.DataFrame],
) -> list[StabilityLegendHandle]:
    """Build stability legend handles for the classifications present in the given dataframes."""
    present_stabilities: set = set()
    for dataframe in fixed_points_dataframes:
        if dataframe is not None and not dataframe.empty:
            present_stabilities |= set(dataframe[Column.FIXED_POINT_STABILITY].unique())

    return [
        StabilityLegendHandle(stability_label=stability)
        for stability in StabilityLabel
        if stability in present_stabilities
    ]
