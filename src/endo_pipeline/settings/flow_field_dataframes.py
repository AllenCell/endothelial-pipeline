from enum import StrEnum

from matplotlib.lines import Line2D

"""Global constants and default settings for dataframe creation and processing in 3D flow field analysis."""

DATAFRAME_MANIFEST_PREFIX_DRIFT: dict[int, str] = {
    1: "drift_vector_field_1d",
    2: "drift_vector_field_2d",
    3: "drift_vector_field_3d",
}
"""Prefix for setting and getting dataframe manifest name for drift dataframes
in n-dimensional flow field analysis."""

FMS_ANNOTATION_NOTES_DRIFT: dict[int, str] = {
    1: "Drift vectors and corresponding grid points for 1D flow field estimation.",
    2: "Drift vectors and corresponding grid points for 2D flow field estimation.",
    3: "Drift vectors and corresponding grid points for 3D flow field estimation.",
}
"""Annotation notes for drift coefficient dataframes uploaded to FMS in
n-dimensional flow field analysis."""

DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS: dict[int, str] = {
    1: "drift_fixed_points_1d",
    2: "drift_fixed_points_2d",
    3: "drift_fixed_points_3d",
}
"""Prefix for setting and getting dataframe manifest name for fixed point dataframes
in n-dimensional flow field analysis."""

FMS_ANNOTATION_NOTES_FIXED_POINTS: dict[int, str] = {
    1: "High-confidence fixed points identified from 1D flow field analysis.",
    2: "High-confidence fixed points identified from 2D flow field analysis.",
    3: "High-confidence fixed points identified from 3D flow field analysis.",
}
"""Annotation notes for fixed point dataframes uploaded to FMS in n-dimensional flow field analysis."""

DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING: str = "bootstrapped_fixed_points"
"""Prefix for setting and getting dataframe manifest name for bootstrapped fixed
point dataframes (3D only)."""

FMS_ANNOTATION_NOTES_BOOTSTRAPPING: str = (
    "Bootstrapped fixed points from 3D flow field analysis, including "
    "confidence intervals and detection rates."
)
"""Annotation notes for bootstrapped fixed point dataframes uploaded to FMS in
3D flow field analysis."""


class StabilityLabel(StrEnum):
    """Fixed point stability classification labels."""

    STABLE = "stable"
    """Label for stable fixed points."""

    SADDLE = "saddle"
    """Label for saddle point fixed points."""

    UNSTABLE = "unstable"
    """Label for unstable fixed points."""

    INDETERMINATE = "indeterminate"
    """Label for fixed points with indeterminate stability."""

    NODE = "node"
    """Label for fixed points classified as nodes (i.e., real eigenvalues with
    the same sign)."""

    SPIRAL = "spiral"
    """Label for fixed points classified as spirals (i.e., complex conjugate
    eigenvalues with nonzero imaginary part)."""


STABILITY_COLOR_DICT: dict[str, str] = {
    StabilityLabel.STABLE: "blue",
    StabilityLabel.SADDLE: "grey",
    StabilityLabel.UNSTABLE: "red",
    StabilityLabel.INDETERMINATE: "khaki",
}
"""Dictionary mapping fixed point stability classification labels to colors for visualization."""

STABILITY_MARKER_DICT: dict[str, str] = {
    StabilityLabel.STABLE: "o",
    StabilityLabel.SADDLE: "^",
    StabilityLabel.UNSTABLE: "s",
    StabilityLabel.INDETERMINATE: "P",
}
"""Dictionary mapping fixed point stability classification labels to marker styles for visualization."""


class StabilityLegendHandle(Line2D):
    """Custom legend handle for fixed point stability classifications in 3D dynamics analysis visualizations."""

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
            marker=marker or STABILITY_MARKER_DICT.get(stability_label, "o"),
            color=face_color or STABILITY_COLOR_DICT.get(stability_label, "gray"),
            markersize=marker_size,
            markeredgecolor=edge_color,
            linestyle="",
        )
