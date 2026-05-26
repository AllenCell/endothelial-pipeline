from enum import StrEnum
from typing import Literal

"""Global constants and default settings for dataframe creation and processing in 3D flow field analysis."""

DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD: str = "drift_vector_field"
"""Prefix for vector field dataframe manifest name."""

FMS_ANNOTATION_NOTES_VECTOR_FIELD: str = (
    "Drift vectors and corresponding grid points for 3D flow field estimation."
)
"""Annotation notes for vector field dataframes uploaded to FMS."""

DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS: str = "drift_fixed_points"
"""Prefix for fixed points dataframe manifest name."""

FMS_ANNOTATION_NOTES_FIXED_POINTS: str = (
    "High-confidence fixed points identified from %dD flow field analysis."
)
"""Annotation notes for fixed point dataframes uploaded to FMS."""

DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING: str = "bootstrapped_fixed_points"
"""Prefix for setting and getting dataframe manifest name for bootstrapped fixed point dataframes."""

GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME: str = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_grid"
"""Dataframe manifest name for grid-based bootstrapping results."""

BOOTSTRAPPING_MANIFEST_NAMES: dict[Literal["grid", "tracked"], str] = {
    "grid": GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME,
}
"""Mapping of crop pattern to bootstrapping dataframe manifest name."""

FMS_ANNOTATION_NOTES_BOOTSTRAPPING: str = (
    "Bootstrapped fixed points from 3D flow field analysis, including "
    "confidence intervals and detection rates."
)
"""Annotation notes for bootstrapped fixed point dataframes uploaded to FMS in 3D flow field analysis."""


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
