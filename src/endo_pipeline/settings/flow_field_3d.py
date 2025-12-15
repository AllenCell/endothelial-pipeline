KERNEL_PARAMS_3D: dict = {
    "bandwidth": 0.125,
    "kernel": "gaussian",
}
"""Default kernel parameters for 3D flow field estimation."""

NUM_BINS_3D: tuple[int, int, int] = (60, 60, 50)
"""Default number of bins for 3D flow field estimation."""

QUIVER_DOWNSAMPLE_FACTOR: int = 2
"""Downsample factor for quiver plots in 3D flow field visualization."""

QUIVER_VECTOR_SCALE: float = 50.0
"""Vector scale for quiver plots in 3D flow field visualization."""

NORMALIZE_QUIVER_VECTORS: bool = True
"""Whether to normalize quiver vectors in 3D flow field visualization."""

LOG_NORM_MAGNITUDES: bool = True
"""Whether to use logarithmic normalization for vector magnitude colormap in 3D flow field visualization."""

CLIP_MAGNITUDES: bool = True
"""Whether to clip vector magnitudes for colormap in 3D flow field visualization."""

CLIP_MIN_MAGNITUDE_PERCENTILE: float | None = 0.1
"""Percentile for clipping minimum vector magnitudes in 3D flow field visualization."""

CLIP_MAX_MAGNITUDE_PERCENTILE: float | None = None
"""Percentile for clipping maximum vector magnitudes in 3D flow field visualization."""

QUIVER_COLORMAP: str = "turbo"
"""Colormap for quiver plots in 3D flow field visualization."""

KDE_CONTOUR_COLORMAP: str = "Greys"
"""Colormap for KDE contours in 3D flow field visualization."""

KDE_CONTOUR_LEVELS: int = 25
"""Number of contour levels for KDE contours in 3D flow field visualization."""

KDE_CONTOUR_OPACITY: float = 0.75
"""Opacity for KDE contours in 3D flow field visualization."""

FLOW_FIELD_X_AXIS_LABEL: str = "PC1"
"""Label for the X axis in the plots of 2D slices of the 3D flow field."""

FLOW_FIELD_Y_AXIS_LABELS: tuple[str, str] = ("PC2", "PC3")
"""Labels for the Y axes in the plots of 2D slices of the 3D flow field."""

TIME_STEP_IN_MINUTES: int = 5
"""Time step in minutes between consecutive time points for flow field estimation."""

INIT_POINT_3D: list = [0.5, 0.0, 0.5]
"""Default initial point for 3D flow field trajectory visualization."""

TRAJECTORY_TIME_SPAN: list[int] = [0, 5000]
"""Default time span for ODE solver in 3D flow field trajectory visualization."""

DATASET_COLLECTION_FOR_3D_DYNAMICS: str = "3d_flow_field_analysis"
"""Default dataset collection name for 3D dynamics analysis."""

OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS: str = "flow_field_3d"
"""Default output folder name for 3D dynamics analysis."""

TRAJECTORY_DICT_FILE_NAME: str = "traj_dict"
"""Default file name for saving trajectory dictionaries."""
