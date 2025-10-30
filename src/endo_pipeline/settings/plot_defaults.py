"""Default settings for plotting functions."""

from endo_pipeline.configs import ShearStressRegime

NUM_BINS_CROP_HIST = 40
"""Number of bins for histograms used in crop montage sampling."""

SHEAR_COLOR_DICT = {
    ShearStressRegime.NO: "tab:green",
    ShearStressRegime.MIN: "tab:orange",
    ShearStressRegime.LOW: "tab:red",
    ShearStressRegime.MEDIUM: "tab:purple",
    ShearStressRegime.HIGH: "tab:cyan",
    ShearStressRegime.MAX: "tab:blue",
}
"""Color dictionary for shear stress levels to color code histogram plots."""

MODEL_QC_SUBPLOT_KWARGS: dict = {"frame_on": False}
"""Default keyword arguments for subplots in model QC plots."""

MODEL_QC_GRIDSPEC_KWARGS: dict = {"wspace": 0.05, "hspace": 0.05}
"""Default keyword arguments for gridspec in model QC plots."""

MODEL_QC_FIG_KWARGS: dict = {"figsize": (6, 6)}
"""Default keyword arguments for figure in model QC plots."""

MODEL_QC_PLOT_DIRECTION: str = "top-down first"
"""Default direction for arranging panels in model QC plots."""
