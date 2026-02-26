"""Reusable helpers for TVL1 optical-flow feature extraction.

These utilities are used by the optical-flow workflow but are kept here so
that downstream analysis scripts can also call them directly (e.g. for
single-position debugging or notebook exploration).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from skimage.registration import optical_flow_tvl1

from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.configs.dataset_config_utils import (
    get_annotated_timepoints_for_position,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.workflow_defaults import OPTICAL_FLOW_BASE_FEATURES

# ---------------------------------------------------------------------------
# Channel-aware intensity percentile
# ---------------------------------------------------------------------------
_CHANNEL_PERCENTILE = {
    "EGFP": 95,  # sparse fluorescence → exclude empty background
    "BF": 0,     # dense texture → keep all pixels
}
_DEFAULT_PERCENTILE = 95

def resolve_percentile(channel: list, explicit: Optional[int] = None) -> int:
    """Return the intensity percentile for thresholding.

    If *explicit* is not None the caller overrode the default → use it.
    Otherwise look up the channel in the built-in table
    (EGFP → 95, BF → 0, default → 95).
    """
    if explicit is not None:
        return explicit
    ch = channel[0] if len(channel) == 1 else None
    return _CHANNEL_PERCENTILE.get(ch, _DEFAULT_PERCENTILE)

def build_optical_flow_feature_cols(max_dt: int) -> list:
    """Return all optical-flow column names for dt = 1..max_dt."""
    return [f"{f}_dt{d}" for d in range(1, max_dt + 1) for f in OPTICAL_FLOW_BASE_FEATURES]


# ---------------------------------------------------------------------------
# Flow statistics
# ---------------------------------------------------------------------------
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
    """Compute summary statistics from a (u, v) flow field.

    Only pixels where at least one of the two frames exceeds *thresh* are
    included.  Returns a flat dict suitable for ``pd.DataFrame(records)``.
    """
    base = {"crop_index": crop_idx, "timepoint": timepoint, "dt": dt}
    mask = (crop0 > thresh) | (crop1 > thresh)
    if not mask.any():
        base.update({f: np.nan for f in OPTICAL_FLOW_BASE_FEATURES})
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
            "optical_flow_mean_angle": float(
                np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
            ),
            "optical_flow_angle_std": float(stats.circstd(ang)),
            "optical_flow_mean_u": float(um.mean()),
            "optical_flow_mean_v": float(vm.mean()),
            "optical_flow_std_u": float(um.std()),
            "optical_flow_std_v": float(vm.std()),
        }
    )
    return base


def compute_crop_flow(
    c0: np.ndarray,
    c1: np.ndarray,
    crop_idx: int,
    tp: int,
    dt: int,
    thresh: float = 0.0,
) -> dict:
    """Run TVL1 optical flow on a single crop pair and return summary stats."""
    v, u = optical_flow_tvl1(c0, c1)
    return flow_stats(u, v, c0, c1, crop_idx, tp, dt, thresh)


# ---------------------------------------------------------------------------
# Timepoint / crop helpers
# ---------------------------------------------------------------------------
def get_valid_timepoints(
    ds_config,
    pos_idx: int,
    annotations_to_exclude: List[TimepointAnnotation],
) -> List[int]:
    """Return sorted timepoints *not* carrying any excluded annotation."""
    bad: set = set()
    for ann in annotations_to_exclude:
        bad.update(
            get_annotated_timepoints_for_position(
                ds_config, pos_idx, annotations=[ann]
            )
        )
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
    sz = (
        int(df[ColumnName.CROP_SIZE_X].iloc[0])
        if ColumnName.CROP_SIZE_X in df.columns
        else 128
    )
    crop_df["end_x"] = crop_df[ColumnName.START_X] + sz
    crop_df["end_y"] = crop_df[ColumnName.START_Y] + sz
    return crop_df


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------
def pivot_flow_records(records: List[dict], max_dt: int) -> pd.DataFrame:
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
        pv.columns = [f"{feat}_dt{int(c)}" for c in pv.columns]
        parts.append(pv)
    return pd.concat(parts, axis=1).reset_index()
