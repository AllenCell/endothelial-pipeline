OUTLIER_THRESHOLD: float = 0.01
"""Percentage to use for thresholding dark and bright gfp and bf outliers."""

PARTIAL_DARK_THRESHOLD: float = 0.004
"""Percentage to use for thresholding partial dark bf outliers."""

GFP_ROLLING_WINDOW: int = 12
"""Number of timepoints to use for rolling window calculation of gfp outliers (1 hour)."""

BF_ROLLING_WINDOW: int = 100
"""Number of z-slices per to use for rolling window calculation of bf outliers (4 timepoints)."""

DEFAULT_MLFLOW_TRACKING_URI: str = "https://production.int.allencell.org/mlflow/"
"""Default MLflow tracking URI for model logging and retrieval."""
