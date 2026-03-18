"""Global constants and default settings for dataframe creation and processing in 3D flow field analysis."""

DATAFRAME_MANIFEST_PREFIX_DRIFT: str = "drift_vector_field"
"""Prefix for setting and getting dataframe manifest name for drift dataframes
in 3D flow field analysis."""

FMS_ANNOTATION_NOTES_DRIFT: str = (
    "Drift vectors and corresponding grid points for 3D flow field estimation."
)
"""Annotation notes for drift coefficient dataframes uploaded to FMS in 3D flow field analysis."""

DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS: str = "drift_fixed_points"
"""Prefix for setting and getting dataframe manifest name for fixed point dataframes
in 3D flow field analysis."""

FMS_ANNOTATION_NOTES_FIXED_POINTS: str = (
    "High-confidence fixed points identified from 3D flow field analysis."
)
"""Annotation notes for fixed point dataframes uploaded to FMS in 3D flow field analysis."""
