"""Settings for 3D flow field estimation and visualization."""

KERNEL_FUNCTION_NAME: str = "gaussian"
"""Default kernel function name for 3D flow field estimation."""

KERNEL_BANDWIDTH: float = 0.2
"""Default kernel bandwidth for 3D flow field estimation."""

BIN_WIDTH_DEFAULTS: tuple[float, float, float] = (0.05, 0.05, 0.05)
"""Default number of bins for 3D flow field estimation."""

PAD_BINS_FLOAT: float = 0.1
"""Percentage of padding to add to the min and max of each axis when creating bins for 3D flow field estimation."""

QUIVER_DOWNSAMPLE_FACTOR: int = 2
"""Downsample factor for quiver plots in 3D flow field visualization."""

QUIVER_VECTOR_SCALE: float = 40.0
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

QUIVER_COLORMAP: str = "crest"
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

NROWS_2D_FLOW_FIELD: int = 2
"""Number of rows for the 2D flow field visualization figure."""

NCOLS_2D_FLOW_FIELD: int = 1
"""Number of columns for the 2D flow field visualization figure."""

FIGSIZE_2D_FLOW_FIELD: tuple[int, int] = (7, 10)
"""Figure size for the 2D flow field visualization figure."""

FIGSIZE_FLOW_FIELD_STACK: tuple[int, int] = (7, 5)
"""Figure size for the flow field stack visualization figure."""

TIME_STEP_IN_MINUTES: int = 5
"""Time step in minutes between consecutive time points for flow field estimation."""

INIT_POINT_3D: tuple[float, float, float] = (1.5, 0.2, -0.5)
"""Default initial point for 3D flow field trajectory visualization."""

TRAJECTORY_TIME_SPAN: tuple[float, float] = (0.0, 5000.0)
"""Default time span for ODE solver in 3D flow field trajectory visualization."""

NUM_INIT_SAMPLES: int = 250
"""Number of sampled initial points for root finding in 3D flow field analysis."""

SAMPLER_RANDOM_SEED: int = 47
"""Random seed for initial point sampling in 3D flow field analysis."""

UPPER_PERCENTILE_FOR_STABLE_FP: float = 95.0
"""Upper percentile threshold for stable fixed point identification in 3D flow field analysis."""

LOWER_PERCENTILE_FOR_STABLE_FP: float = 5.0
"""Lower percentile threshold for stable fixed point identification in 3D flow field analysis."""

DATASET_COLLECTION_FOR_3D_DYNAMICS: str = "3d_flow_field_analysis"
"""Default dataset collection name for 3D dynamics analysis."""

DATAFRAME_MANIFEST_PREFIX_DRIFT: str = "flow_field_drift"
"""Prefix for setting and getting dataframe manifest name for drift dataframes
in 3D flow field analysis."""

FMS_ANNOTATION_NOTES_DRIFT: str = "Drift coefficients for 3D flow field estimation."
"""Annotation notes for drift coefficient dataframes uploaded to FMS in 3D flow field analysis."""

DATAFRAME_MANIFEST_PREFIX_GRID: str = "flow_field_grid"
"""Prefix for setting and getting dataframe manifest name for grid dataframes in
3D flow field analysis."""

FMS_ANNOTATION_NOTES_GRID: str = "Grid point coordinates for 3D flow field estimation."
"""Annotation notes for grid point dataframes uploaded to FMS in 3D flow field analysis."""

DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS: str = "flow_field_fixed_points"
"""Prefix for setting and getting dataframe manifest name for fixed point dataframes
in 3D flow field analysis."""

FMS_ANNOTATION_NOTES_FIXED_POINTS: str = (
    "Stable fixed points identified from 3D flow field analysis."
)
"""Annotation notes for fixed point dataframes uploaded to FMS in 3D flow field analysis."""

DATAFRAME_OUTPUT_DIR: str = "flow_field_3d_dataframes"
"""Directory for storing 3D flow field dataframes."""
