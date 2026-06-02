"""Classes and methods for visualizing fixed points."""

from matplotlib.lines import Line2D

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
