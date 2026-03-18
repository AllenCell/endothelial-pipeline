"""Reusable helpers for TVL1 optical-flow feature extraction.

Provides all compute, I/O, and visualization utilities used by the
``compute_optical_flow_feats`` workflow.  Downstream scripts and notebooks
can also import directly from this package or its submodules.

Submodules
----------
config         Channel-aware parameter resolution, annotation exclusion, constants.
compute        Core numerical work — TVL1, statistics, block averaging.
dataframe      DataFrame wrangling — crop grids, pivoting, column names.
io             Disk writes, FMS uploads, manifest registration.
visualization  Matplotlib diagnostic plots (imported lazily when needed).
"""

from .compute import (
    _block_average_flow,
    compute_crop_flow,
    compute_flow_statistics,
    compute_image_pair_flow,
    compute_tvl1,
)
from .config import (
    COHERENCE_BOX_SIZES,
    default_annotations_to_exclude,
    resolve_attachment,
    resolve_percentile,
)
from .dataframe import (
    build_crop_grid,
    build_optical_flow_feature_cols,
    pivot_flow_records,
)
from .io import save_and_upload, save_parquet
from .visualization import plot_demo_summary

__all__ = [
    # config
    "COHERENCE_BOX_SIZES",
    "resolve_percentile",
    "resolve_attachment",
    "default_annotations_to_exclude",
    # compute
    "compute_tvl1",
    "compute_flow_statistics",
    "compute_crop_flow",
    "compute_image_pair_flow",
    "_block_average_flow",
    # dataframe
    "build_optical_flow_feature_cols",
    "build_crop_grid",
    "pivot_flow_records",
    # io
    "save_parquet",
    "save_and_upload",
    # visualization
    "plot_demo_summary",
]
