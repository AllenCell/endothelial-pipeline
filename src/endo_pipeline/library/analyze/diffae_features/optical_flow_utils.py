"""
Reusable helpers for TVL1 optical-flow feature extraction.

These utilities are used by the optical-flow workflow but are kept here so
that downstream scripts can also call them directly (e.g. for
single-position debugging or notebook exploration).

"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from skimage.registration import optical_flow_tvl1

from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.configs.dataset_config_utils import (
    DatasetConfig,
    get_annotated_timepoints_for_position,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.workflow_defaults import (
    OPTICAL_FLOW_BASE_FEATURES,
    OPTICAL_FLOW_CHANNEL_PERCENTILE,
)

logger = logging.getLogger(__name__)


def resolve_percentile(channel: list, explicit: int | None = None) -> int:
    """Return the intensity percentile for thresholding.

    If *explicit* is None, look up the channel in the built-in table
    (EGFP → 95, BF → 0), else use that value!
    """
    if len(channel) != 1:
        raise ValueError(f"Optical flow operates on a single channel, got {channel}.")
    if explicit is not None:
        return explicit
    ch = channel[0]
    if ch not in OPTICAL_FLOW_CHANNEL_PERCENTILE:
        raise ValueError(
            f"Unknown channel {ch!r}. "
            f"Supported channels: {sorted(OPTICAL_FLOW_CHANNEL_PERCENTILE)}. "
            "Pass an explicit percentile to override."
        )
    return OPTICAL_FLOW_CHANNEL_PERCENTILE[ch]


def build_optical_flow_feature_cols(max_dt: int) -> list:
    """Return all optical-flow column names for dt = 1..max_dt."""
    return [f"{f}_dt{d}" for d in range(1, max_dt + 1) for f in OPTICAL_FLOW_BASE_FEATURES]


# Flow statistics
def flow_stats(
    u: np.ndarray,
    v: np.ndarray,
    crop0: np.ndarray,
    crop1: np.ndarray,
    crop_idx: int,
    timepoint: int,
    dt: int,
    thresh: float,
) -> dict:
    """Compute summary statistics from a 2-D optical-flow field (u, v).

    Pixels are included in all statistics only when the intensity in
    *either* ``crop0`` or ``crop1`` exceeds ``thresh``, discarding
    background regions where flow estimates are unreliable.

    Parameters
    ----------
    u
        Horizontal (x-axis) flow component, shape ``(H, W)``, in pixels
        per unit time.
    v
        Vertical (y-axis) flow component, shape ``(H, W)``, in pixels
        per unit time.
    crop0
        Intensity image/crop at a given timepoint, shape ``(H, W)``.
    crop1
        Intensity image/crop at the next timepoint, shape ``(H, W)``.
    crop_idx
        Index identifying which spatial crop this flow field belongs to.
    timepoint
        Index of the frame/crop in the sequence.
    dt
        Temporal stride between the two frames (in frame units).
        Used to normalise velocities to a per-frame rate when needed.
    thresh
        Intensity threshold for foreground masking.  A pixel is included
        when ``max(crop0[i,j], crop1[i,j]) > thresh``.

    Returns
    -------
        Flat dictionary of scalar statistics (mean speed, mean u/v,
        percentiles, masked pixel count, etc.) together with the
        identifying fields ``crop_idx`` and ``timepoint``.  One dict
        corresponds to one row in a table built from a list
        of records.
    """
    base: dict[str, int | float] = {"crop_index": crop_idx, "timepoint": timepoint, "dt": dt}
    mask = (crop0 > thresh) | (crop1 > thresh)
    if not mask.any():
        logger.debug(
            "No foreground pixels above thresh=%.3g for crop_idx=%d, timepoint=%d; returning NaNs.",
            thresh,
            crop_idx,
            timepoint,
        )
        base.update(dict.fromkeys(OPTICAL_FLOW_BASE_FEATURES, np.nan))
        return base

    sp = np.sqrt(u[mask] ** 2 + v[mask] ** 2)
    ang = np.arctan2(v[mask], u[mask])
    um, vm = u[mask], v[mask]

    nz = sp > 0
    muv = (
        float(np.sqrt(np.mean(um[nz] / sp[nz]) ** 2 + np.mean(vm[nz] / sp[nz]) ** 2))
        if nz.any()
        else 0.0
    )
    base.update(
        {
            "optical_flow_mean_speed": float(sp.mean()),
            "optical_flow_mean_unit_vector": muv,
            "optical_flow_std_speed": float(sp.std()),
            "optical_flow_mean_angle": float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())),
            "optical_flow_angle_std": float(stats.circstd(ang)),
            "optical_flow_mean_u": float(um.mean()),
            "optical_flow_mean_v": float(vm.mean()),
            "optical_flow_std_u": float(um.std()),
            "optical_flow_std_v": float(vm.std()),
        }
    )
    return base


def compute_tvl1(
    f0: np.ndarray,
    f1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TVL1 optical flow on two 2-D frames.

    Returns ``(u, v)`` — horizontal and vertical flow components — so
    that all TVL1 calls in the project go through a single helper.

    Parameters
    ----------
    f0
        Intensity frame at a given timepoint, shape ``(H, W)``.
    f1
        Intensity frame at the successive timepoint, shape ``(H, W)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(u, v)`` flow arrays, each shaped ``(H, W)``.
    """
    v, u = optical_flow_tvl1(f0, f1)
    return u, v


def compute_crop_flow(
    c0: np.ndarray,
    c1: np.ndarray,
    crop_idx: int,
    tp: int,
    dt: int,
    thresh: float = 0.0,
) -> dict:
    """Run TVL1 optical flow on a single crop pair and return summary statistics.

    Wraps :func:`compute_tvl1` and delegates to :func:`flow_stats` for
    masked aggregation.  Intended to be called in parallel across crops
    and timepoints.

    Parameters
    ----------
    c0
        Intensity image/crop at a given timepoint, shape ``(H, W)``.
    c1
        Intensity image/crop at the next timepoint, shape ``(H, W)``.
    crop_idx
        Index identifying this spatial crop.
    tp
        Timepoint index of ``c0`` in the source sequence.
    dt
        Temporal stride between ``c0`` and ``c1`` (in frame units).
    thresh
        Intensity threshold for foreground masking (default ``0.0``).
        Pixels where both frames are at or below this value are excluded
        from all statistics.

    Returns
    -------
        Flat statistics dict
    """
    u, v = compute_tvl1(c0, c1)
    return flow_stats(u, v, c0, c1, crop_idx, tp, dt, thresh)


def compute_image_pair_flow(
    f0: np.ndarray,
    f1: np.ndarray,
    sy: np.ndarray,
    ey: np.ndarray,
    sx: np.ndarray,
    ex: np.ndarray,
    crop_indices: np.ndarray,
    t0: int,
    dt: int,
    thresh: float,
) -> list[dict]:
    """Run TVL1 on a full-resolution frame pair, then compute per-crop stats.

    This is the *image-scope* flow strategy: TVL1 runs once on the full
    image and the resulting flow field is sliced per crop before computing
    summary statistics.  Compared to the crop-scope approach
    (:func:`compute_crop_flow`), this is faster and avoids boundary
    artefacts at crop edges.

    Parameters
    ----------
    f0, f1
        Full-resolution intensity frames, shape ``(H, W)``.
    sy, ey, sx, ex
        Arrays of per-crop spatial bounds (start/end y/x), length
        ``n_crops``.
    crop_indices
        Array of crop-index identifiers, length ``n_crops``.
    t0
        Timepoint index of ``f0``.
    dt
        Temporal stride between ``f0`` and ``f1``.
    thresh
        Intensity threshold for foreground masking.

    Returns
    -------
        One statistics dict per crop (see :func:`flow_stats`).
    """
    u, v = compute_tvl1(f0, f1)
    n_crops = len(crop_indices)
    return [
        flow_stats(
            u[sy[i] : ey[i], sx[i] : ex[i]],
            v[sy[i] : ey[i], sx[i] : ex[i]],
            f0[sy[i] : ey[i], sx[i] : ex[i]],
            f1[sy[i] : ey[i], sx[i] : ex[i]],
            int(crop_indices[i]),
            t0,
            dt,
            thresh,
        )
        for i in range(n_crops)
    ]


# ---------------------------------------------------------------------------
# Timepoint / crop helpers
# ---------------------------------------------------------------------------
def get_valid_timepoints(
    ds_config: DatasetConfig,
    pos_idx: int,
    annotations_to_exclude: list[TimepointAnnotation],
) -> list[int]:
    """Return sorted timepoints *not* carrying any excluded annotation."""
    bad: set[int] = set()
    for ann in annotations_to_exclude:
        bad.update(get_annotated_timepoints_for_position(ds_config, pos_idx, annotations=[ann]))
    return sorted(set(range(ds_config.duration)) - bad)


def build_crop_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique crop index with spatial bounds.

    Adds ``end_x`` and ``end_y`` columns derived from
    ``ColumnName.START_X / START_Y`` + crop size.
    """
    crop_df = (
        df[[ColumnName.START_X, ColumnName.START_Y, ColumnName.CROP_INDEX]]
        .drop_duplicates(subset=[ColumnName.CROP_INDEX])
        .sort_values(by=[ColumnName.START_Y, ColumnName.START_X])
        .reset_index(drop=True)
    )
    sz = int(df[ColumnName.CROP_SIZE_X].iloc[0]) if ColumnName.CROP_SIZE_X in df.columns else 128
    crop_df["end_x"] = crop_df[ColumnName.START_X] + sz
    crop_df["end_y"] = crop_df[ColumnName.START_Y] + sz
    return crop_df


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------
def pivot_flow_records(records: list[dict]) -> pd.DataFrame:
    """Pivot a list of flow-stat dicts into a wide DataFrame.

    Returns one row per ``(crop_index, timepoint)`` with columns named
    ``{feature}_dt{n}``.
    """
    df = pd.DataFrame(records)
    parts = []
    for feat in OPTICAL_FLOW_BASE_FEATURES:
        pv = df.pivot_table(
            index=["crop_index", "timepoint"],
            columns="dt",
            values=feat,
            aggfunc="first",
        )
        pv.columns = pd.Index([f"{feat}_dt{int(c)}" for c in pv.columns])
        parts.append(pv)
    return pd.concat(parts, axis=1).reset_index()
