"""Default settings for plotting functions."""

from matplotlib.lines import Line2D

from endo_pipeline.configs import ShearStressRegime
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel

CROP_HIST_BIN_WIDTH: float = 0.15
"""Bin width used in crop montage sampling."""

SHEAR_COLOR_DICT = {
    (ShearStressRegime.NO,): "tab:green",
    (ShearStressRegime.MIN,): "tab:orange",
    (ShearStressRegime.LOW,): "tab:red",
    (ShearStressRegime.MEDIUM,): "tab:purple",
    (ShearStressRegime.HIGH,): "tab:cyan",
    (ShearStressRegime.MAX,): "tab:blue",
    (ShearStressRegime.MIN, ShearStressRegime.MAX): "tab:brown",
    (ShearStressRegime.MAX, ShearStressRegime.MIN): "tab:olive",
}
"""Color dictionary for shear stress levels to color code histogram plots."""

FIXED_POINT_PLOT_STYLE: dict[str, dict[str, str]] = {
    StabilityLabel.STABLE: {"color": "blue", "marker": "o"},
    StabilityLabel.SADDLE: {"color": "grey", "marker": "^"},
    StabilityLabel.UNSTABLE: {"color": "red", "marker": "s"},
    StabilityLabel.INDETERMINATE: {"color": "khaki", "marker": "P"},
}
"""Dictionary mapping fixed point stability classification labels to plotting styles for visualizations."""


class StabilityLegendHandle(Line2D):
    """Custom legend handle for fixed point stability classifications in dynamics analysis visualizations."""

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
            marker=marker or FIXED_POINT_PLOT_STYLE.get(stability_label, {}).get("marker", "o"),
            color=face_color
            or FIXED_POINT_PLOT_STYLE.get(stability_label, {}).get("color", "gray"),
            markersize=marker_size,
            markeredgecolor=edge_color,
            linestyle="",
        )


MODEL_QC_SUBPLOT_KWARGS: dict = {"frame_on": False}
"""Default keyword arguments for subplots in model QC plots."""

MODEL_QC_GRIDSPEC_KWARGS: dict = {"wspace": 0.03, "hspace": 0.03}
"""Default keyword arguments for gridspec in model QC plots."""

MODEL_QC_FIG_KWARGS: dict = {"figsize": (9, 6)}
"""Default keyword arguments for figure in model QC plots."""

MODEL_QC_PLOT_DIRECTION: str = "top-down first"
"""Default direction for arranging panels in model QC plots."""
