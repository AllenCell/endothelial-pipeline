"""Default settings for plotting functions."""

from collections import namedtuple
from math import pi

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

MarkerStyle = namedtuple("MarkerStyle", ["marker", "color"])
"""Named tuple for defining marker style in plots."""

FIXED_POINT_PLOT_STYLE: dict[str, MarkerStyle] = {
    StabilityLabel.STABLE: MarkerStyle(marker="o", color="blue"),
    StabilityLabel.SADDLE: MarkerStyle(marker="^", color="grey"),
    StabilityLabel.UNSTABLE: MarkerStyle(marker="s", color="red"),
    StabilityLabel.INDETERMINATE: MarkerStyle(marker="P", color="khaki"),
}
"""Dictionary mapping fixed point stability classification labels to plotting styles for visualizations."""

MODEL_QC_SUBPLOT_KWARGS: dict = {"frame_on": False}
"""Default keyword arguments for subplots in model QC plots."""

MODEL_QC_GRIDSPEC_KWARGS: dict = {"wspace": 0.03, "hspace": 0.03}
"""Default keyword arguments for gridspec in model QC plots."""

MODEL_QC_FIG_KWARGS: dict = {"figsize": (9, 6)}
"""Default keyword arguments for figure in model QC plots."""

MODEL_QC_PLOT_DIRECTION: str = "top-down first"
"""Default direction for arranging panels in model QC plots."""

POLAR_THETA_RANGE: tuple[float, float] = (-pi / 6, (5 * pi / 6) + 0.1)
"""Default theta range for polar plots of angles in radians."""
