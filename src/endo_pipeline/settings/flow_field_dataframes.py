"""Global constants and default settings for dataframe creation and processing in 3D flow field analysis."""

from enum import StrEnum

FMS_ANNOTATION_NOTES_VECTOR_FIELD: str = (
    "Drift vectors and corresponding grid points for 3D flow field estimation."
)
"""Annotation notes for vector field dataframes uploaded to FMS."""

FMS_ANNOTATION_NOTES_FIXED_POINTS: str = (
    "High-confidence fixed points identified from %dD flow field analysis."
)
"""Annotation notes for fixed point dataframes uploaded to FMS."""


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
