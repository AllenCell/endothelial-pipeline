"""Default settings for plotting functions."""

NUM_BINS_CROP_HIST = 40
"""Number of bins for histograms used in crop montage sampling."""


SHEAR_COLOR_NO = "tab:green"
"""No shear stress color for plotting."""

SHEAR_COLOR_MIN = "tab:orange"
"""Minimum shear stress color for plotting."""

SHEAR_COLOR_LOW = "tab:red"
"""Low shear stress color for plotting."""

SHEAR_COLOR_MEDIUM = "tab:purple"
"""Medium shear stress color for plotting."""

SHEAR_COLOR_HIGH = "tab:cyan"
"""High shear stress color for plotting."""

SHEAR_COLOR_MAX = "tab:blue"
"""Maximum shear stress color for plotting."""

SHEAR_COLOR_DICT = {
    "no": SHEAR_COLOR_NO,
    "min": SHEAR_COLOR_MIN,
    "low": SHEAR_COLOR_LOW,
    "medium": SHEAR_COLOR_MEDIUM,
    "high": SHEAR_COLOR_HIGH,
    "max": SHEAR_COLOR_MAX,
}
