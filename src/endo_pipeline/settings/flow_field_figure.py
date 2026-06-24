"""
Global settings for flow field paper figure (results from dynamics analysis
at low and high shear stress).
"""

from typing import Any

from endo_pipeline.settings.column_names import ColumnName as Column

GRIDSPEC_KWARGS: dict[str, Any] = {"wspace": 0.1, "hspace": 0.1}
"""Gridspec kwargs used for figure panels."""

XLABEL_KWARGS: dict[str, Any] = {"labelpad": 2}
"""X-axis label kwargs used for figure panels."""

YLABEL_KWARGS: dict[str, Any] = {"labelpad": 2}
"""Y-axis label kwargs used for figure panels."""

AXES_LIMITS_2D: dict[Column.DiffAEData, tuple[float, float]] = {
    Column.DiffAEData.POLAR_RADIUS: (0.2, 1.8),
    Column.DiffAEData.PC3_FLIPPED: (-1.05, 1.05),
}
"""Axes limits for 2D plots in polar radius and rho variables."""

NULLCLINE_STYLES_2D: dict[Column.DiffAEData, tuple[int, tuple[int, int]] | str] = {
    Column.DiffAEData.POLAR_RADIUS: "dashed",
    Column.DiffAEData.PC3_FLIPPED: (0, (1, 1)),  # dense dotted
}
"""Nullcline styles for 2D plots in polar radius and rho variables."""

SUPP_FIG_TARGET_POINT = (1.0, -0.1)
"""Highlighted target point for schematic panels in the supplementary figure
detailing flow field construction."""

SUPP_FIG_ZOOM_FACTOR = 3.5
"""Factor to determine the zoomed-in region around the target point for the
schematic panels in the supplementary figure detailing flow field construction."""
