"""Reusable helpers for TVL1 optical-flow feature extraction.

Provides all compute, I/O, and visualization utilities used by the
``compute_optical_flow_feats`` workflow.  Downstream scripts and notebooks
can also import directly from this package or its submodules.

Submodules
----------
params         Channel-aware parameter resolution and annotation exclusion.
compute        Core numerical work — TVL1, statistics, block averaging.
dataframe      DataFrame wrangling — crop grids, pivoting, column names.
io             Disk writes, FMS uploads, manifest registration.
visualization  Matplotlib diagnostic plots.
"""

from .compute import compute_crop_flow, compute_image_pair_flow
from .dataframe import (
    build_crop_grid,
    build_ema_stems,
    build_optical_flow_feature_cols,
    pivot_flow_records,
)
from .params import resolve_attachment, resolve_percentile
from .visualization import plot_demo_summary, plot_tracked_crop_coherence_timeseries

__all__ = [
    "build_crop_grid",
    "build_ema_stems",
    "build_optical_flow_feature_cols",
    "compute_crop_flow",
    "compute_image_pair_flow",
    "pivot_flow_records",
    "plot_demo_summary",
    "plot_tracked_crop_coherence_timeseries",
    "resolve_attachment",
    "resolve_percentile",
]
