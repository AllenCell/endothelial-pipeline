KERNEL_PARAMS_3D: dict = {
    "bandwidth": 0.125,
    "kernel": "gaussian",
}
"""Default kernel parameters for 3D flow field estimation."""

NUM_BINS_3D: tuple[int, int, int] = (40, 40, 40)
"""Default number of bins for 3D flow field estimation."""

QUIVER_DOWNSAMPLE_FACTOR: int = 2
"""Downsample factor for quiver plots in 3D flow field visualization."""

QUIVER_VECTOR_SCALE: float = 50.0
"""Vector scale for quiver plots in 3D flow field visualization."""

QUIVER_OVERLAY_COLOR: str = "dimgrey"
"""Overlay color for quiver plots with scatter plot overlay in 3D flow field visualization."""

NORMALIZE_QUIVER_VECTORS: bool = True
"""Whether to normalize quiver vectors in 3D flow field visualization."""

TIME_STEP_IN_MINUTES: int = 5
"""Time step in minutes between consecutive time points for flow field estimation."""

INIT_POINT_3D: list = [0.5, 0.0, -1.0]
"""Default initial point for 3D flow field trajectory visualization."""

TRAJECTORY_TIME_SPAN: list[int] = [0, 5000]
"""Default time span for ODE solver in 3D flow field trajectory visualization."""

DATASET_COLLECTION_FOR_3D_DYNAMICS: str = "3d_flow_field_analysis"
"""Default dataset collection name for 3D dynamics analysis."""

OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS: str = "flow_field_3d"
"""Default output folder name for 3D dynamics analysis."""

TRAJECTORY_DICT_FILE_NAME: str = "traj_dict"
"""Default file name for saving trajectory dictionaries."""
