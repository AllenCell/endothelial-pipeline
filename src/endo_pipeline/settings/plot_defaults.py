"""Default settings for plotting functions."""

from endo_pipeline.configs import ShearStressRegime

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

MODEL_QC_SUBPLOT_KWARGS: dict = {"frame_on": False}
"""Default keyword arguments for subplots in model QC plots."""

MODEL_QC_GRIDSPEC_KWARGS: dict = {"wspace": 0.03, "hspace": 0.03}
"""Default keyword arguments for gridspec in model QC plots."""

MODEL_QC_FIG_KWARGS: dict = {"figsize": (9, 6)}
"""Default keyword arguments for figure in model QC plots."""

MODEL_QC_PLOT_DIRECTION: str = "top-down first"
"""Default direction for arranging panels in model QC plots."""

DRIFT_CONTOUR_VMIN: float = -0.25
"""Minimum value for contour plots of drift components."""

DRIFT_CONTOUR_VMAX: float = 0.25
"""Maximum value for contour plots of drift components."""

DRIFT_CONTOUR_LEVELS: int = 50
"""Number of contour levels to use in contour plots of drift components."""
