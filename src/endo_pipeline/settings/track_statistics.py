"""Global constants for the trajectory statistics workflow."""

from endo_pipeline.settings.literal_types import PatchTypeLiteral

BIN_WIDTH_FOR_AVERAGE = 0.1
"""Bin width for KDE histogram of per-trajectory average values in each
coordinate."""

BIN_WIDTH_FOR_VARIANCE = 0.02
"""Bin width for KDE histogram of per-trajectory variance values in each
coordinate."""

BIN_PAD_FOR_VARIANCE: float = 0.05
"""Padding to add to the left and right of the min and max variance values
when computing bins for KDE histogram of per-trajectory variance values in each
coordinate."""

NUM_POINTS_SMOOTH_KDE = 2000
"""Number of points at which to interpolate KDE via spline."""

AXES_YLIM_FOR_AVERAGE: tuple[float, float] = (-0.05, 1.75)
"""Y-axis limits for KDE plot of per-trajectory average values in each
coordinate."""

AXES_XLIM_FOR_VARIANCE: tuple[float, float] = (-0.05, 0.8)
"""X-axis limits for KDE plot of per-trajectory variance values in each
coordinate."""

AXES_YLIM_FOR_VARIANCE: tuple[float, float] = (-0.05, 15)
"""Y-axis limits for KDE plot of per-trajectory variance values in each
coordinate."""

KDE_LINESTYLE_DICT: dict[PatchTypeLiteral, str] = {
    "grid_based": "-",
    "cell_centered": "--",
}
"""Line style kwargs for KDE plots of per-trajectory average and variance
values in each coordinate, by patch type."""

KDE_LABEL_DICT: dict[PatchTypeLiteral, str] = {
    "grid_based": "grid",
    "cell_centered": "tracked (bootstrap mean)",
}

KDE_LINE_KWARGS: dict[str, str | float] = {
    "color": "k",
    "linewidth": 2,
}
"""Line kwargs for KDE plots of per-trajectory average and variance values in
each coordinate."""

CI_FILL_OPACITY: float = 0.15
"""Opacity for confidence interval fill in plots of per-trajectory average and
variance distributions."""
