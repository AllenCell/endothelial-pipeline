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
