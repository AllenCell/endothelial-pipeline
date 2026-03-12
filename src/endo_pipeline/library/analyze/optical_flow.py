"""Reusable helpers for TVL1 optical-flow feature extraction.

Provides all compute, I/O, and visualization utilities consumed by the
``compute_optical_flow_feats`` workflow.  Downstream scripts and notebooks
can also import directly from this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from skimage.registration import optical_flow_tvl1

from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.workflow_defaults import (
    OPTICAL_FLOW_BASE_FEATURES,
    OPTICAL_FLOW_CHANNEL_ATTACHMENT,
    OPTICAL_FLOW_CHANNEL_PERCENTILE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COHERENCE_BOX_SIZES: tuple[int, ...] = (
    1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
)
"""Non-overlapping box sizes (in pixels) for multi-scale coherence."""

_DEMO_SCAN_N_CROPS: int = 6
_DEMO_SCAN_N_PAIRS: int = 10
_QUIVER_GRID_DIVISIONS: int = 8


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _block_average_flow(
    u: np.ndarray,
    v: np.ndarray,
    box: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average (u, v) flow vectors within non-overlapping box*box blocks.

    Parameters
    ----------
    u, v
        Flow components, shape ``(H, W)``.
    box
        Side length of each square block.  When ``box == 1`` the arrays
        are returned unchanged.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Block-averaged ``(u, v)`` arrays, shape ``(H // box, W // box)``.
    """
    if box == 1:
        return u, v
    H, W = u.shape
    Ht = (H // box) * box
    Wt = (W // box) * box
    u_b = u[:Ht, :Wt].reshape(Ht // box, box, Wt // box, box).mean(axis=(1, 3))
    v_b = v[:Ht, :Wt].reshape(Ht // box, box, Wt // box, box).mean(axis=(1, 3))
    return u_b, v_b


def _speed_weighted_circ_std(
    angles: np.ndarray,
    speeds: np.ndarray,
) -> float:
    """Compute speed-weighted circular standard deviation.

    Parameters
    ----------
    angles
        1-D array of angles in radians.
    speeds
        1-D array of speeds (used as weights).  Must be non-negative.

    Returns
    -------
    float
        Speed-weighted circular standard deviation (radians).  Returns
        ``nan`` if total weight is zero.
    """
    total_w = speeds.sum()
    if total_w == 0:
        return np.nan
    C = np.sum(speeds * np.cos(angles)) / total_w
    S = np.sum(speeds * np.sin(angles)) / total_w
    R_bar = min(np.sqrt(C**2 + S**2), 1.0)  # clamp for numerical safety
    return float(np.sqrt(-2.0 * np.log(R_bar)))


# ---------------------------------------------------------------------------
# Channel-aware parameter resolution
# ---------------------------------------------------------------------------
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


def resolve_attachment(channel: list, explicit: float | None = None) -> float:
    """Return the TVL1 attachment (lambda) for a given channel.

    If *explicit* is None, look up the channel in the built-in table
    (EGFP → 7.5, BF → 25.0), else use the given value.

    Parameters
    ----------
    channel
        Single-element list with channel name.
    explicit
        Override value.  When provided, the channel table is ignored.

    Returns
    -------
    float
        Attachment value to pass to :func:`compute_tvl1`.
    """
    if len(channel) != 1:
        raise ValueError(f"Optical flow operates on a single channel, got {channel}.")
    if explicit is not None:
        return float(explicit)
    ch = channel[0]
    if ch not in OPTICAL_FLOW_CHANNEL_ATTACHMENT:
        raise ValueError(
            f"Unknown channel {ch!r}. "
            f"Supported channels: {sorted(OPTICAL_FLOW_CHANNEL_ATTACHMENT)}. "
            "Pass an explicit attachment to override."
        )
    return OPTICAL_FLOW_CHANNEL_ATTACHMENT[ch]


# ---------------------------------------------------------------------------
# Feature column helpers
# ---------------------------------------------------------------------------
def build_optical_flow_feature_cols(max_dt: int) -> list:
    """Return all optical-flow column names for dt = 1..max_dt."""
    return [f"{f}_dt{d}" for d in range(1, max_dt + 1) for f in OPTICAL_FLOW_BASE_FEATURES]


# ---------------------------------------------------------------------------
# Flow statistics
# ---------------------------------------------------------------------------
def compute_flow_statistics(
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
    thresh
        Intensity threshold for foreground masking.  A pixel is included
        when ``max(crop0[i,j], crop1[i,j]) > thresh``.

    Returns
    -------
        Flat dictionary of scalar statistics together with the
        identifying fields ``crop_idx`` and ``timepoint``.
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

    # --- Multi-scale coherence ---
    u_2d = u.copy()
    v_2d = v.copy()
    u_2d[~mask] = 0.0
    v_2d[~mask] = 0.0

    for box in _COHERENCE_BOX_SIZES:
        ub, vb = _block_average_flow(u_2d, v_2d, box)
        sp_b = np.sqrt(ub**2 + vb**2)
        ang_b = np.arctan2(vb, ub)
        nz_b = sp_b > 0
        key = f"optical_flow_angle_std_box{box}"
        if nz_b.any():
            base[key] = float(stats.circstd(ang_b[nz_b]))
        else:
            base[key] = np.nan

    return base


# ---------------------------------------------------------------------------
# TVL1 wrappers
# ---------------------------------------------------------------------------
def compute_tvl1(
    f0: np.ndarray,
    f1: np.ndarray,
    attachment: float = 7.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TVL1 optical flow on two 2-D frames.

    Returns ``(u, v)`` — horizontal and vertical flow components.

    Parameters
    ----------
    f0
        Intensity frame at a given timepoint, shape ``(H, W)``.
    f1
        Intensity frame at the successive timepoint, shape ``(H, W)``.
    attachment
        TVL1 attachment (lambda).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(u, v)`` flow arrays, each shaped ``(H, W)``.
    """
    v, u = optical_flow_tvl1(f0, f1, attachment=attachment)
    return u, v


def compute_crop_flow(
    c0: np.ndarray,
    c1: np.ndarray,
    crop_idx: int,
    tp: int,
    dt: int,
    thresh: float = 0.0,
    attachment: float = 7.5,
) -> dict:
    """Run TVL1 optical flow on a single crop pair and return summary statistics.

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
    attachment
        TVL1 attachment (lambda).  See :func:`compute_tvl1`.

    Returns
    -------
        Flat statistics dict.
    """
    u, v = compute_tvl1(c0, c1, attachment=attachment)
    return compute_flow_statistics(u, v, c0, c1, crop_idx, tp, dt, thresh)


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
    attachment: float = 7.5,
) -> list[dict]:
    """Run TVL1 on a full-resolution frame pair, then compute per-crop stats.

    This is the *image-scope* flow strategy: TVL1 runs once on the full
    image and the resulting flow field is sliced per crop before computing
    summary statistics.

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
    attachment
        TVL1 attachment (lambda).  See :func:`compute_tvl1`.

    Returns
    -------
        One statistics dict per crop (see :func:`compute_flow_statistics`).
    """
    u, v = compute_tvl1(f0, f1, attachment=attachment)
    n_crops = len(crop_indices)
    return [
        compute_flow_statistics(
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
# Crop helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Annotation exclusion
# ---------------------------------------------------------------------------
def default_annotations_to_exclude(
    include_cell_piling: bool = False,
    include_pre_steady_state: bool = False,
) -> list[TimepointAnnotation]:
    """Build the default timepoint-annotation exclusion list.

    Quality annotations (scope errors, temporary artifacts, XY/Z shifts,
    unfed) are **always** excluded — computing optical flow against
    corrupted frames would contaminate neighbouring good frames.  This
    keeps :func:`~endo_pipeline.configs.get_unannotated_timepoints_for_position`
    in sync with the dataframe filter applied by
    :func:`~endo_pipeline.library.analyze.diffae_dataframe_utils.get_dataframe_for_dynamics_workflows`.

    The two lifecycle annotations (``CELL_PILING``,
    ``NOT_STEADY_STATE``) are controlled by boolean flags.

    Parameters
    ----------
    include_cell_piling
        When ``False``, timepoints annotated as
        :attr:`TimepointAnnotation.CELL_PILING` are excluded.
    include_pre_steady_state
        When ``False``, timepoints annotated as
        :attr:`TimepointAnnotation.NOT_STEADY_STATE` are excluded.

    Returns
    -------
        Annotations whose timepoints should be filtered out.
    """
    excl: list[TimepointAnnotation] = [
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
        TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
        TimepointAnnotation.GFP_SCOPE_ERROR,
        TimepointAnnotation.UNFED,
        TimepointAnnotation.XY_SHIFT,
        TimepointAnnotation.Z_SHIFT,
    ]
    if not include_pre_steady_state:
        excl.append(TimepointAnnotation.NOT_STEADY_STATE)
    if not include_cell_piling:
        excl.append(TimepointAnnotation.CELL_PILING)
    return excl


# ---------------------------------------------------------------------------
# FMS upload & manifest registration
# ---------------------------------------------------------------------------
def save_parquet(dataset: str, df: pd.DataFrame) -> Path:
    """Write a per-dataset optical-flow dataframe to parquet.

    Parameters
    ----------
    dataset
        Dataset name, used to construct the output filename.
    df
        DataFrame to persist.

    Returns
    -------
        Absolute path to the written parquet file.
    """
    from endo_pipeline.io import get_output_path

    out = get_output_path("optical_flow") / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{dataset}_optical_flow_manifest.parquet"
    df.to_parquet(p, index=False)
    logger.info("Saved parquet: %s", p)
    return p


def save_and_upload(dataset: str, df: pd.DataFrame) -> str:
    """Save a parquet, upload it to FMS, and register it in the manifest.

    Persists *df* via :func:`save_parquet`, uploads the file to FMS
    then creates or updates the optical-flow dataframe manifest so
    that downstream consumers can locate the result by dataset name.

    Parameters
    ----------
    dataset
        Dataset name used for the parquet filename and manifest key.
    df
        Optical-flow feature DataFrame to persist.

    Returns
    -------
        FMS file identifier for the uploaded parquet.
    """
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import build_fms_annotations, upload_file_to_fms
    from endo_pipeline.manifests import (
        DataframeLocation,
        DataframeManifest,
        create_dataframe_manifest,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.workflow_defaults import DEFAULT_OPTICAL_FLOW_MANIFEST_NAME

    logger.info("Saving and uploading results for %s", dataset)
    p = save_parquet(dataset, df)
    cfg = load_dataset_config(dataset)
    fms_id = upload_file_to_fms(p, build_fms_annotations(cfg), "parquet")
    logger.info("Uploaded to FMS: %s", fms_id)
    try:
        manifest = load_dataframe_manifest(DEFAULT_OPTICAL_FLOW_MANIFEST_NAME)
    except Exception:
        try:
            manifest = create_dataframe_manifest(DEFAULT_OPTICAL_FLOW_MANIFEST_NAME, __file__)
        except Exception:
            manifest = DataframeManifest(
                name=DEFAULT_OPTICAL_FLOW_MANIFEST_NAME, workflow=Path(__file__).stem
            )
    if manifest.locations is None:
        manifest.locations = {}
    manifest.locations[dataset] = DataframeLocation(fmsid=fms_id)
    save_dataframe_manifest(manifest)
    return fms_id


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_demo_summary(
    cache: dict[int, np.ndarray],
    crop_grid: pd.DataFrame,
    ds_name: str,
    pos_str: str,
    thresh: float,
    out_dir: Path,
    channel: list[str],
    flow_scope: str,
    attachment: float = 7.5,
) -> None:
    """Produce a multi-crop diagnostic figure (up to 3 rows x 4 cols).

    Scans a subsample of (crop, timepoint) pairs, picks up to three
    representative pairs — most coherent, median, and most
    incoherent — then plots for each:
      (a) Red/Green composite of crop at t0 (red) and t1 (green).
      (b) Quiver plot of TVL1 flow.
      (c) Angle histogram (masked pixels only).
      (d) sigma_theta vs block size bar chart.

    Skips plotting entirely if the cache contains fewer than 2 frames
    or the subsample scan yields fewer than 2 valid records.

    Parameters
    ----------
    cache
        Mapping from timepoint index to its 2-D intensity frame,
        shape ``(H, W)``.
    crop_grid
        One row per spatial crop with columns for crop index and
        start/end x/y coordinates (see :func:`build_crop_grid`).
    ds_name
        Dataset name, used in the figure title and output filename.
    pos_str
        Position identifier (e.g. ``"P0"``).
    thresh
        Intensity threshold for foreground masking.
    out_dir
        Directory where the PNG figure is saved.
    channel
        Imaging channel name(s) (e.g. ``["BF"]``).
    flow_scope
        Flow computation strategy (``"image"`` or ``"crop"``).
    attachment
        TVL1 attachment (lambda) value.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy import stats as sp_stats

    plt.style.use("endo_pipeline.figure")

    # Scan a subsample of crops x timepoints to find coherent/incoherent flow
    sorted_tp = sorted(cache.keys())
    if len(sorted_tp) < 2:
        logger.warning("Only %d cached frame(s) — skipping demo plot", len(sorted_tp))
        return

    cids = crop_grid[ColumnName.CROP_INDEX].values
    sx_arr = crop_grid[ColumnName.START_X].values.astype(int)
    sy_arr = crop_grid[ColumnName.START_Y].values.astype(int)
    ex_arr = crop_grid["end_x"].values.astype(int)
    ey_arr = crop_grid["end_y"].values.astype(int)

    crop_step = max(1, len(cids) // _DEMO_SCAN_N_CROPS)
    scan_cids = range(0, len(cids), crop_step)

    all_pairs = [(sorted_tp[i], sorted_tp[i + 1]) for i in range(len(sorted_tp) - 1)]
    pair_step = max(1, len(all_pairs) // _DEMO_SCAN_N_PAIRS)
    scan_pairs = all_pairs[::pair_step]

    records: list[dict] = []
    _image_flow_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for ci_idx in scan_cids:
        _sx, _sy = int(sx_arr[ci_idx]), int(sy_arr[ci_idx])
        _ex, _ey = int(ex_arr[ci_idx]), int(ey_arr[ci_idx])
        _cidx = int(cids[ci_idx])
        for t0, t1 in scan_pairs:
            f0, f1 = cache[t0], cache[t1]
            c0, c1 = f0[_sy:_ey, _sx:_ex], f1[_sy:_ey, _sx:_ex]
            if flow_scope == "image":
                if (t0, t1) not in _image_flow_cache:
                    _image_flow_cache[(t0, t1)] = compute_tvl1(f0, f1, attachment=attachment)
                uf_full, vf_full = _image_flow_cache[(t0, t1)]
                uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]
            else:
                uf, vf = compute_tvl1(c0, c1, attachment=attachment)
            ang = np.arctan2(vf, uf)
            mask = (c0 > thresh) | (c1 > thresh)
            cstd = float(sp_stats.circstd(ang[mask])) if mask.any() else float("nan")

            records.append(
                {
                    "ci_idx": ci_idx,
                    "crop": _cidx,
                    "t0": t0,
                    "t1": t1,
                    "sx": _sx,
                    "sy": _sy,
                    "ex": _ex,
                    "ey": _ey,
                    "circ_std": cstd,
                }
            )

    scan_df = pd.DataFrame(records).dropna(subset=["circ_std"])
    if len(scan_df) < 2:
        logger.warning("Scan produced <2 valid records — skipping demo plot")
        return

    scan_df = scan_df.sort_values("circ_std").reset_index(drop=True)
    n_scanned = len(scan_df)

    if n_scanned >= 3:
        picks = [
            (scan_df.iloc[0], r"COHERENT (low $\sigma_{\theta}$)"),
            (scan_df.iloc[n_scanned // 2], r"MEDIAN $\sigma_{\theta}$"),
            (scan_df.iloc[-1], r"INCOHERENT (high $\sigma_{\theta}$)"),
        ]
    else:
        picks = [
            (scan_df.iloc[0], r"COHERENT (low $\sigma_{\theta}$)"),
            (scan_df.iloc[-1], r"INCOHERENT (high $\sigma_{\theta}$)"),
        ]

    logger.info(
        "Demo scan: %d valid pairs, picked %d crops for plot",
        n_scanned,
        len(picks),
    )
    for row, label in picks:
        logger.info(
            "%s: sigma_theta=%.4f (crop %d, t=%d->%d)",
            label.split("(")[-1].rstrip(")$"),
            row["circ_std"],
            int(row["crop"]),
            int(row["t0"]),
            int(row["t1"]),
        )

    # Helper: plot one row (4 panels)
    def _plot_row(axes, row, label):  # noqa: C901
        t0, t1 = int(row["t0"]), int(row["t1"])
        _sx, _sy = int(row["sx"]), int(row["sy"])
        _ex, _ey = int(row["ex"]), int(row["ey"])
        _cidx = int(row["crop"])
        cy, cx = _ey - _sy, _ex - _sx

        f0, f1 = cache[t0], cache[t1]
        c0, c1 = f0[_sy:_ey, _sx:_ex], f1[_sy:_ey, _sx:_ex]
        if flow_scope == "image":
            uf_full, vf_full = compute_tvl1(f0, f1, attachment=attachment)
            uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]
        else:
            uf, vf = compute_tvl1(c0, c1, attachment=attachment)
        sp = np.sqrt(uf**2 + vf**2)
        ang = np.arctan2(vf, uf)
        mask = (c0 > thresh) | (c1 > thresh)
        cstd = float(sp_stats.circstd(ang[mask])) if mask.any() else 0.0

        def _norm(im):
            lo, hi = np.percentile(im, [2, 99.5])
            return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)

        # (a) Red/Green composite
        ax = axes[0]
        ax.set_facecolor("white")
        rgb = np.zeros((cy, cx, 3), dtype=np.float32)
        rgb[..., 0] = _norm(c0)
        rgb[..., 1] = _norm(c1)
        ax.imshow(rgb, origin="upper")
        ax.set_title(f"(a) Composite  crop {_cidx}\n{label}", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"$\\sigma_{{\\theta}}$ = {cstd:.4f}", fontsize=10, fontstyle="italic")
        ax.legend(
            handles=[
                Patch(facecolor="red", label=f"t={t0}"),
                Patch(facecolor="green", label=f"t={t1}"),
                Patch(facecolor="yellow", label="overlap"),
            ],
            fontsize=6,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            framealpha=0.7,
            borderaxespad=0,
        )
        ax.tick_params(labelsize=7)

        # (b) Quiver
        ax = axes[1]
        ax.set_facecolor("white")
        step = max(1, cy // _QUIVER_GRID_DIVISIONS)
        Y, X = np.mgrid[0:cy:step, 0:cx:step]
        sp_sub = sp[::step, ::step]
        med_sp = float(np.median(sp_sub[sp_sub > 0])) if (sp_sub > 0).any() else 1.0
        q_scale = med_sp / (step * 0.6) if med_sp > 0 else 1.0
        ax.quiver(
            X,
            Y,
            uf[::step, ::step],
            vf[::step, ::step],
            sp_sub,
            cmap="autumn",
            clim=[0, np.percentile(sp, 97)],
            angles="xy",
            scale_units="xy",
            scale=q_scale,
            width=0.008,
            headwidth=4,
            headlength=5,
            minshaft=1.5,
            alpha=0.85,
        )
        ax.set_xlim(0, cx)
        ax.set_ylim(cy, 0)
        ax.set_aspect("equal")
        ax.set_title(f"(b) Quiver  t={t0}\u2192{t1}", fontsize=9)
        ax.tick_params(labelsize=7)

        # (c) Angle histogram
        ax = axes[2]
        ax.set_facecolor("white")
        if mask.any():
            ang_m = ang[mask]
            ax.hist(ang_m, bins=72, color="salmon", edgecolor="white", linewidth=0.3, density=True)
            cmean = float(np.arctan2(np.sin(ang_m).mean(), np.cos(ang_m).mean()))
            ax.axvline(cmean, color="red", ls="--", lw=1.5, label=f"circ \u03bc = {cmean:.2f}")
            ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("\u03b8 (rad)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(
            f"(c) $\\theta$ distribution  $\\sigma_{{\\theta}}$ = {cstd:.4f} rad", fontsize=9
        )
        ax.tick_params(labelsize=7)

        # (d) Multi-scale coherence bar chart
        ax = axes[3]
        ax.set_facecolor("white")
        box_sizes = list(_COHERENCE_BOX_SIZES)
        box_vals = []
        for b in box_sizes:
            ub, vb = _block_average_flow(uf, vf, b)
            sp_b = np.sqrt(ub**2 + vb**2)
            ang_b = np.arctan2(vb, ub)
            nz_b = sp_b > 0
            box_vals.append(
                float(sp_stats.circstd(ang_b[nz_b])) if nz_b.any() else float("nan")
            )
        x_pos = np.arange(len(box_sizes))
        colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(box_sizes)))
        bars = ax.bar(x_pos, box_vals, color=colors, edgecolor="white", linewidth=0.5)
        for bar_rect, val in zip(bars, box_vals, strict=True):
            if not np.isnan(val):
                ax.text(
                    bar_rect.get_x() + bar_rect.get_width() / 2,
                    val + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(b) for b in box_sizes], fontsize=7)
        ax.set_xlabel("Block size (px)", fontsize=8)
        ax.set_ylabel(r"$\sigma_{\theta}$ (rad)", fontsize=8)
        ax.set_title(r"(d) $\sigma_{\theta}$ vs block size", fontsize=9)
        ax.tick_params(labelsize=7)

    # Build the Nx4 figure (N = 2 or 3 depending on scan count)
    n_rows = len(picks)
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(24, 4.5 * n_rows), facecolor="white", squeeze=False
    )

    for row_idx, (row, label) in enumerate(picks):
        _plot_row(axes[row_idx], row, label)

    fig.suptitle(
        f"Coherent vs Incoherent : {ds_name} / {pos_str}  [{', '.join(channel)}]  (scope={flow_scope})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir
        / f"demo_coherent_vs_incoherent_{ds_name}_{pos_str}_{'_'.join(channel)}_{flow_scope}.png",
        dpi=300,
        facecolor="white",
    )
    logger.info("Saved coherent-vs-incoherent figure to %s", out_dir)
    plt.close(fig)
