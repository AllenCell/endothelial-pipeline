"""Settings for perturbation datasets: knockout (KO) datasets and their isogenic controls."""

PERTURBATION_COLOR: str = "tab:pink"
"""Default color for KO datasets and isogenic controls in plots."""

KO_PLOT_MARKERS: tuple[str, str, str] = ("X", "D", "*")
"""Default marker style for KO datasets."""

ISOGENIC_CONTROL_PLOT_MARKERS: tuple[str, str] = ("^", "s")

KO_CELL_LINE: str = "AICS-177 cl. 26"
"""Default cell line name for KO datasets."""

ISOGENIC_CONTROL_CELL_LINE: str = "AICS-177"
"""Default cell line name for isogenic control datasets."""
