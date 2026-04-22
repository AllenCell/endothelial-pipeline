"""Supplementary figure: optical-flow coherent vs incoherent example panels.

Renders a 2x2 figure showing one COHERENT (high R-bar) and one INCOHERENT
(low R-bar) crop/timepoint pair from a single dataset+position.  Each row
contains the (a) red/green composite of consecutive frames and the
(b) TVL1 quiver plot.

The first run does an *exhaustive* scan across all positions x all crops
x all consecutive frame pairs (within ``n_frames``) and caches the
``rbar`` table to a parquet under
``results/supp_fig_optical_flow/scan_cache/`` (no timestamp, so it
persists).  Subsequent runs simply load that table and pick fresh
COHERENT/INCOHERENT rows -- no TVL1 needed for selection, only the
2 final image-scope quivers.

Reuses the TVL1 + R-bar logic from
``endo_pipeline.library.analyze.optical_flow``.  Saves the figure under
``results/<date>/supp_fig_optical_flow/`` as both PNG and SVG, following
the standard endo_pipeline figures convention.
"""

# %%
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from tqdm.auto import tqdm

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_unannotated_timepoints_for_position,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path, load_dataframe, load_image
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.analyze.optical_flow import (
    build_crop_grid,
    default_annotations_to_exclude,
    resolve_attachment,
    resolve_percentile,
)
from endo_pipeline.library.analyze.optical_flow.compute import compute_tvl1
from endo_pipeline.manifests import get_zarr_location_for_position, load_dataframe_manifest
from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL, DIMENSION_ORDER
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.optical_flow import (
    DEMO_MAX_FRAMES,
    OPTICAL_FLOW_COLUMNS_TO_COMPUTE,
    QUIVER_GRID_DIVISIONS,
)
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

plt.style.use("endo_pipeline.figure")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_feature_df(dataset_name: str) -> pd.DataFrame:
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"
    manifest = load_dataframe_manifest(f"{base_name}_pca")
    cols = list(OPTICAL_FLOW_COLUMNS_TO_COMPUTE["grid"])
    return load_dataframe(manifest.locations[dataset_name], delay=True)[cols].compute()


def _build_frame_cache(
    dataset_name: str,
    position: int,
    channel: Literal["BF", "EGFP"],
    level: int,
    n_frames: int,
    annotations_to_exclude: list[TimepointAnnotation],
    df_dataset: pd.DataFrame | None = None,
    required_timepoints: list[int] | None = None,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Replicate the (channel-aware) z-projection + normalization used by
    ``compute_optical_flow_feats``, returning a ``{tp: 2D frame}`` cache
    and the position's crop grid.

    ``required_timepoints`` are forced into the cache regardless of
    ``n_frames`` -- useful for manual picks at large ``t0``.
    """
    import dask.array as da

    is_bf = channel == "BF"
    dataset_config = load_dataset_config(dataset_name)
    if df_dataset is None:
        df_dataset = _load_feature_df(dataset_name)

    all_valid = list(
        get_unannotated_timepoints_for_position(
            dataset_config, position, annotations_to_exclude
        )
    )
    valid_timepoints = list(all_valid[:n_frames])
    if required_timepoints:
        for t in required_timepoints:
            t = int(t)
            if t not in valid_timepoints:
                valid_timepoints.append(t)
    df_position = df_dataset[
        (df_dataset[Column.POSITION] == position)
        & (df_dataset[Column.TIMEPOINT].isin(valid_timepoints))
    ].copy()
    if df_position.empty:
        raise RuntimeError(
            f"No rows for {dataset_name} pos={position} in feature dataframe"
        )
    crop_grid = build_crop_grid(df_position)

    zarr_path = get_zarr_location_for_position(dataset_config, position)
    image_dask = load_image(zarr_path, channels=[channel], level=level, compute=False)
    z_axis = DIMENSION_ORDER.index("Z")
    z_projection = (
        da.log(image_dask.std(axis=z_axis) + 1e-12)
        if is_bf
        else image_dask.max(axis=z_axis)
    )

    needed_indices = sorted(valid_timepoints)
    needed_frames = z_projection[needed_indices, 0].compute(scheduler="threads")
    cache: dict[int, np.ndarray] = {}
    for j, t in enumerate(needed_indices):
        frame = needed_frames[j].astype(np.float32, copy=False)
        if is_bf:
            lo, hi = np.percentile(frame, [0.1, 99.9])
            frame = np.clip(frame, lo, hi)
            std = frame.std()
            frame = (frame - frame.mean()) / (std if std > 0 else 1.0)
        else:
            lo, hi = np.percentile(frame, [10, 98])
            frame = np.clip(frame, lo, hi)
            frame = (frame - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        cache[t] = frame
    return cache, crop_grid


def _compute_intensity_threshold(
    cache: dict[int, np.ndarray], intensity_pctl: int
) -> float:
    if intensity_pctl <= 0:
        return -float("inf")
    return float(
        np.percentile(
            np.concatenate([f.ravel()[::10] for f in cache.values()]),
            intensity_pctl,
        )
    )


def _rbar(uf: np.ndarray, vf: np.ndarray, c0: np.ndarray, c1: np.ndarray, thresh: float) -> float:
    sp = np.sqrt(uf**2 + vf**2)
    mask = (c0 > thresh) | (c1 > thresh)
    nz = mask & (sp > 0)
    if not nz.any():
        return float("nan")
    u = uf[nz] / sp[nz]
    v = vf[nz] / sp[nz]
    return float(np.sqrt(u.mean() ** 2 + v.mean() ** 2))


def _scan_position_full(
    cache: dict[int, np.ndarray],
    crop_grid: pd.DataFrame,
    thresh: float,
    selection_scope: Literal["image", "crop"],
    attachment: float,
) -> pd.DataFrame:
    """Exhaustive (every crop x every consecutive pair) R-bar scan for one position."""
    sorted_tp = sorted(cache.keys())
    pairs = [(sorted_tp[i], sorted_tp[i + 1]) for i in range(len(sorted_tp) - 1)]

    cids = crop_grid[Column.CROP_INDEX].values
    sx = crop_grid[Column.DiffAEData.START_X].values.astype(int)
    sy = crop_grid[Column.DiffAEData.START_Y].values.astype(int)
    ex = crop_grid["end_x"].values.astype(int)
    ey = crop_grid["end_y"].values.astype(int)

    records: list[dict] = []
    if selection_scope == "image":
        # Compute TVL1 once per pair on the full image, then slice per crop.
        for t0, t1 in pairs:
            uf_full, vf_full = compute_tvl1(cache[t0], cache[t1], attachment=attachment)
            for i, cid in enumerate(cids):
                c0 = cache[t0][sy[i] : ey[i], sx[i] : ex[i]]
                c1 = cache[t1][sy[i] : ey[i], sx[i] : ex[i]]
                uf = uf_full[sy[i] : ey[i], sx[i] : ex[i]]
                vf = vf_full[sy[i] : ey[i], sx[i] : ex[i]]
                records.append(
                    {
                        "crop": int(cid),
                        "t0": int(t0),
                        "t1": int(t1),
                        "sx": int(sx[i]),
                        "sy": int(sy[i]),
                        "ex": int(ex[i]),
                        "ey": int(ey[i]),
                        "rbar": _rbar(uf, vf, c0, c1, thresh),
                    }
                )
    else:
        # crop scope: TVL1 per (crop, pair) on the small patch.
        for t0, t1 in pairs:
            for i, cid in enumerate(cids):
                c0 = cache[t0][sy[i] : ey[i], sx[i] : ex[i]]
                c1 = cache[t1][sy[i] : ey[i], sx[i] : ex[i]]
                uf, vf = compute_tvl1(c0, c1, attachment=attachment)
                records.append(
                    {
                        "crop": int(cid),
                        "t0": int(t0),
                        "t1": int(t1),
                        "sx": int(sx[i]),
                        "sy": int(sy[i]),
                        "ex": int(ex[i]),
                        "ey": int(ey[i]),
                        "rbar": _rbar(uf, vf, c0, c1, thresh),
                    }
                )
    return pd.DataFrame(records).dropna(subset=["rbar"])


def _scan_cache_path(
    dataset_name: str,
    channel: str,
    n_frames: int,
    selection_scope: str,
    level: int,
) -> Path:
    cache_dir = get_output_path(
        "supp_fig_optical_flow", "scan_cache", include_timestamp=False
    )
    return (
        cache_dir
        / f"scan_{dataset_name}_{channel}_n{n_frames}_lvl{level}_{selection_scope}.parquet"
    )


def _load_or_build_full_scan(
    dataset_name: str,
    channel: Literal["BF", "EGFP"],
    n_frames: int,
    level: int,
    selection_scope: Literal["image", "crop"],
    attachment: float,
    intensity_pctl: int,
    annotations_to_exclude: list[TimepointAnnotation],
    positions: list[int] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Load cached exhaustive R-bar scan for *dataset_name* (one row per
    position x crop x consecutive pair) or build & cache it on first run.

    Cached key includes channel, n_frames, level, and selection_scope -- so
    changing any of those triggers a fresh scan.
    """
    cache_path = _scan_cache_path(dataset_name, channel, n_frames, selection_scope, level)
    if cache_path.exists() and not force:
        logger.info("Loading cached optical-flow scan from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info(
        "No cached scan -- building exhaustive R-bar scan for %s "
        "(channel=%s n_frames=%d scope=%s)",
        dataset_name,
        channel,
        n_frames,
        selection_scope,
    )
    df_dataset = _load_feature_df(dataset_name)
    all_positions = sorted(df_dataset[Column.POSITION].unique().tolist())
    if positions:
        all_positions = [p for p in all_positions if p in positions]

    parts: list[pd.DataFrame] = []
    for position in tqdm(all_positions, desc=f"Scanning {dataset_name} positions"):
        try:
            cache, crop_grid = _build_frame_cache(
                dataset_name=dataset_name,
                position=position,
                channel=channel,
                level=level,
                n_frames=n_frames,
                annotations_to_exclude=annotations_to_exclude,
                df_dataset=df_dataset,
            )
        except RuntimeError as exc:
            logger.warning("Skipping pos=%d (%s)", position, exc)
            continue
        thresh = _compute_intensity_threshold(cache, intensity_pctl)
        pos_df = _scan_position_full(cache, crop_grid, thresh, selection_scope, attachment)
        pos_df["position"] = int(position)
        parts.append(pos_df)
        # release frame memory between positions
        cache.clear()

    if not parts:
        raise RuntimeError(
            f"Exhaustive scan produced no rows for dataset {dataset_name}"
        )
    full_df = (
        pd.concat(parts, ignore_index=True)
        .sort_values("rbar", ascending=False)
        .reset_index(drop=True)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)
    logger.info("Cached scan -> %s (%d rows)", cache_path, len(full_df))
    return full_df


def _pick_row_in_range(
    scan_df: pd.DataFrame, target_range: tuple[float, float]
) -> pd.Series:
    """Pick the scanned row whose R-bar is closest to the midpoint of
    *target_range* (preferring rows that fall inside the range)."""
    lo, hi = float(target_range[0]), float(target_range[1])
    mid = 0.5 * (lo + hi)
    in_range = scan_df[(scan_df["rbar"] >= lo) & (scan_df["rbar"] <= hi)]
    pool = in_range if not in_range.empty else scan_df
    idx = (pool["rbar"] - mid).abs().idxmin()
    return pool.loc[idx]


def _resolve_manual_pick(
    spec: dict,
    mode: Literal["max", "min"],
    cache: dict[int, np.ndarray],
    crop_grid: pd.DataFrame,
    thresh: float,
    attachment: float,
) -> pd.Series:
    """Resolve a manually-specified pick.

    *spec* must contain ``position`` and ``t0`` (and optionally ``t1``,
    defaulting to ``t0+1``).  Crop selection precedence:

    1. ``crop`` -- forced crop index.
    2. ``row`` + ``col`` -- 1-indexed row/column in the sorted crop grid
       (rows top-to-bottom, columns left-to-right).
    3. Otherwise -- TVL1 is run on the (t0, t1) pair and the max
       (``mode='max'``) or min (``mode='min'``) R-bar crop is returned.
    """
    position = int(spec["position"])
    t0 = int(spec["t0"])
    t1 = int(spec.get("t1", t0 + 1))
    forced_crop = spec.get("crop")
    forced_row = spec.get("row")
    forced_col = spec.get("col")

    # Sort grid into a stable (row, col) layout based on START_Y / START_X.
    grid_sorted = crop_grid.copy()
    sy_unique = sorted(grid_sorted[Column.DiffAEData.START_Y].unique().tolist())
    sx_unique = sorted(grid_sorted[Column.DiffAEData.START_X].unique().tolist())
    sy_rank = {v: i for i, v in enumerate(sy_unique)}
    sx_rank = {v: i for i, v in enumerate(sx_unique)}
    grid_sorted["_row"] = grid_sorted[Column.DiffAEData.START_Y].map(sy_rank)
    grid_sorted["_col"] = grid_sorted[Column.DiffAEData.START_X].map(sx_rank)

    # Resolve forced crop from row/col if requested.
    if forced_crop is None and forced_row is not None and forced_col is not None:
        target = grid_sorted[
            (grid_sorted["_row"] == int(forced_row) - 1)
            & (grid_sorted["_col"] == int(forced_col) - 1)
        ]
        if target.empty:
            raise RuntimeError(
                f"No crop at row={forced_row} col={forced_col} for pos={position} "
                f"(grid is {len(sy_unique)}x{len(sx_unique)})"
            )
        forced_crop = int(target.iloc[0][Column.CROP_INDEX])

    cids = grid_sorted[Column.CROP_INDEX].values
    sx = grid_sorted[Column.DiffAEData.START_X].values.astype(int)
    sy = grid_sorted[Column.DiffAEData.START_Y].values.astype(int)
    ex = grid_sorted["end_x"].values.astype(int)
    ey = grid_sorted["end_y"].values.astype(int)

    f0, f1 = cache[t0], cache[t1]
    uf_full, vf_full = compute_tvl1(f0, f1, attachment=attachment)

    rows: list[dict] = []
    for i, cid in enumerate(cids):
        c0 = f0[sy[i] : ey[i], sx[i] : ex[i]]
        c1 = f1[sy[i] : ey[i], sx[i] : ex[i]]
        uf = uf_full[sy[i] : ey[i], sx[i] : ex[i]]
        vf = vf_full[sy[i] : ey[i], sx[i] : ex[i]]
        rows.append(
            {
                "position": position,
                "crop": int(cid),
                "t0": t0,
                "t1": t1,
                "sx": int(sx[i]),
                "sy": int(sy[i]),
                "ex": int(ex[i]),
                "ey": int(ey[i]),
                "rbar": _rbar(uf, vf, c0, c1, thresh),
            }
        )
    pair_df = pd.DataFrame(rows).dropna(subset=["rbar"])
    if pair_df.empty:
        raise RuntimeError(
            f"No valid R-bar values for pos={position} t={t0}->{t1}"
        )
    if forced_crop is not None:
        sel = pair_df[pair_df["crop"] == int(forced_crop)]
        if sel.empty:
            raise RuntimeError(
                f"Forced crop {forced_crop} not in grid for pos={position}"
            )
        return sel.iloc[0]
    idx = pair_df["rbar"].idxmax() if mode == "max" else pair_df["rbar"].idxmin()
    return pair_df.loc[idx]


def _plot_pair(
    axes,
    row: pd.Series,
    label: str,
    cache: dict[int, np.ndarray],
    thresh: float,
    flow_scope: str,
    attachment: float,
) -> None:
    """Plot one (composite, quiver) pair for a single scan row."""
    t0, t1 = int(row["t0"]), int(row["t1"])
    _sx, _sy = int(row["sx"]), int(row["sy"])
    _ex, _ey = int(row["ex"]), int(row["ey"])
    cy, cx = _ey - _sy, _ex - _sx

    f0, f1 = cache[t0], cache[t1]
    c0, c1 = f0[_sy:_ey, _sx:_ex], f1[_sy:_ey, _sx:_ex]
    if flow_scope == "image":
        uf_full, vf_full = compute_tvl1(f0, f1, attachment=attachment)
        uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]
    else:
        uf, vf = compute_tvl1(c0, c1, attachment=attachment)
    sp = np.sqrt(uf**2 + vf**2)
    mask = (c0 > thresh) | (c1 > thresh)
    nz = mask & (sp > 0)
    if nz.any():
        u = uf[nz] / sp[nz]
        v = vf[nz] / sp[nz]
        rbar_val = float(np.sqrt(u.mean() ** 2 + v.mean() ** 2))
    else:
        rbar_val = 0.0

    def _norm(im: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(im, [2, 99.5])
        return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)

    # (a) Composite
    ax = axes[0]
    ax.set_facecolor("white")
    rgb = np.zeros((cy, cx, 3), dtype=np.float32)
    rgb[..., 0] = _norm(c0)
    rgb[..., 1] = _norm(c1)
    ax.imshow(rgb, origin="upper")
    ax.set_title(f"(a) Composite\n{label}", fontweight="bold")
    ax.set_ylabel(f"$\\bar{{R}}$ = {rbar_val:.3f}", fontstyle="italic")
    ax.legend(
        handles=[
            Patch(facecolor="red", label=f"t={t0}"),
            Patch(facecolor="green", label=f"t={t1}"),
            Patch(facecolor="yellow", label="overlap"),
        ],
        loc="upper right",
        framealpha=0.0,
        labelcolor="white",
        handlelength=0.9,
        handleheight=0.9,
        borderpad=0.3,
        borderaxespad=0.3,
        labelspacing=0.3,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # (b) Quiver
    ax = axes[1]
    ax.set_facecolor("white")
    step = max(1, cy // QUIVER_GRID_DIVISIONS)
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
        alpha=0.9,
    )
    ax.set_xlim(0, cx)
    ax.set_ylim(cy, 0)
    ax.set_aspect("equal")
    ax.set_title(f"(b) Quiver  t={t0}{Unicode.RIGHT_ARROW}{t1}")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Hard-coded manual picks (set by user from the front-end screenshots).
# Each dict needs at minimum ``position`` and ``t0``; ``t1`` defaults to
# ``t0+1``.  ``dataset`` overrides the workflow ``dataset_name``.
# Crop selection: forced ``crop`` index > 1-indexed ``row``/``col`` in the
# sorted crop grid > auto (max/min R-bar).
DEFAULT_MANUAL_PICKS: dict[str, dict] = {
    "COHERENT": {
        "dataset": "20250409_20X",
        "position": 2,
        "t0": 150,
        "t1": 151,
        "row": 5,
        "col": 4,
    },
    "INCOHERENT": {
        "dataset": "20251001_20X",
        "position": 1,
        "t0": 198,
        "t1": 199,
        "row": 2,
        "col": 5,
    },
}


def main(
    dataset_name: str = "20251001_20X",
    channel: Literal["BF", "EGFP"] = "BF",
    selection_scope: Literal["image", "crop"] = "crop",
    plot_scope: Literal["image", "crop"] = "image",
    level: int = DIFFAE_ZARR_RESOLUTION_LEVEL,
    n_frames: int = DEMO_MAX_FRAMES,
    high_rbar_range: tuple[float, float] = (0.7, 0.95),
    low_rbar_range: tuple[float, float] = (0.1, 0.35),
    figure_size: tuple[float, float] = (5.0, 5.0),
    scan_positions: list[int] | None = None,
    rebuild_scan: bool = False,
    manual_picks: dict[str, dict] | None = None,
) -> None:
    """Build the supplementary figure for optical-flow coherence.

    On first invocation an exhaustive R-bar scan is built across all
    positions x all crops x all consecutive frame pairs (within
    ``n_frames``) for the given dataset/channel/scope/level, and cached
    to a parquet file under ``results/supp_fig_optical_flow/scan_cache/``
    (no timestamp).  Subsequent runs simply load the cached table and
    pick fresh COHERENT/INCOHERENT rows.

    Two scopes are kept independent:

    * *selection_scope* (default ``"crop"``) -- TVL1 scope for the
      cached scan.  Crop scope is much faster (TVL1 on 128x128 patches).
    * *plot_scope* (default ``"image"``) -- TVL1 scope used to render
      the two final quiver panels.  Image scope avoids edge artefacts
      since flow is computed on the full frame and then sliced.

    Parameters
    ----------
    scan_positions
        Optional list of positions to include in the scan (only used
        when building the cache for the first time).
    rebuild_scan
        If True, ignore any cached scan and recompute it.
    """
    annotations_to_exclude = default_annotations_to_exclude(
        include_cell_piling=False, include_pre_steady_state=False
    )
    intensity_pctl = resolve_percentile(channel, None)
    attachment = resolve_attachment(channel)
    use_manual = manual_picks is not None or DEFAULT_MANUAL_PICKS
    if manual_picks is None:
        manual_picks = DEFAULT_MANUAL_PICKS

    logger.info(
        "Building supplementary optical-flow figure: ds=%s ch=%s "
        "selection_scope=%s plot_scope=%s n_frames=%d manual=%s",
        dataset_name,
        channel,
        selection_scope,
        plot_scope,
        n_frames,
        bool(use_manual),
    )

    # ------------------------------------------------------------------
    # Resolve picks (manual fast path vs cached global scan)
    # ------------------------------------------------------------------
    if use_manual:
        # Allow per-pick dataset override; build frame caches keyed by
        # (dataset, position) and load only the timepoint window needed.
        df_cache: dict[str, pd.DataFrame] = {}
        per_key_cache: dict[
            tuple[str, int],
            tuple[dict[int, np.ndarray], pd.DataFrame, float],
        ] = {}
        for spec in manual_picks.values():
            ds = str(spec.get("dataset", dataset_name))
            pos = int(spec["position"])
            key = (ds, pos)
            if key in per_key_cache:
                continue
            t0 = int(spec["t0"])
            t1 = int(spec.get("t1", t0 + 1))
            if ds not in df_cache:
                df_cache[ds] = _load_feature_df(ds)
            cache, crop_grid = _build_frame_cache(
                dataset_name=ds,
                position=pos,
                channel=channel,
                level=level,
                n_frames=n_frames,
                annotations_to_exclude=annotations_to_exclude,
                df_dataset=df_cache[ds],
                required_timepoints=[t0, t1],
            )
            thresh = _compute_intensity_threshold(cache, intensity_pctl)
            per_key_cache[key] = (cache, crop_grid, thresh)

        def _resolve(spec: dict, mode: Literal["max", "min"]) -> pd.Series:
            ds = str(spec.get("dataset", dataset_name))
            pos = int(spec["position"])
            cache, grid, thresh = per_key_cache[(ds, pos)]
            row = _resolve_manual_pick(spec, mode, cache, grid, thresh, attachment)
            row["dataset"] = ds
            return row

        high_row = _resolve(manual_picks["COHERENT"], "max")
        low_row = _resolve(manual_picks["INCOHERENT"], "min")
        # downstream pos_cache wants (cache, thresh) keyed by (dataset, position)
        pos_cache_simple = {k: (c, t) for k, (c, _g, t) in per_key_cache.items()}
    else:
        df_dataset = _load_feature_df(dataset_name)
        scan_df = _load_or_build_full_scan(
            dataset_name=dataset_name,
            channel=channel,
            n_frames=n_frames,
            level=level,
            selection_scope=selection_scope,
            attachment=attachment,
            intensity_pctl=intensity_pctl,
            annotations_to_exclude=annotations_to_exclude,
            positions=scan_positions,
            force=rebuild_scan,
        )
        if len(scan_df) < 2:
            raise RuntimeError("Cached scan has <2 rows -- cannot build figure")
        high_row = _pick_row_in_range(scan_df, high_rbar_range)
        high_row["dataset"] = dataset_name
        low_row = _pick_row_in_range(scan_df, low_rbar_range)
        low_row["dataset"] = dataset_name

    picks = [
        (high_row, r"COHERENT (high $\bar{R}$)", "COHERENT"),
        (low_row, r"INCOHERENT (low $\bar{R}$)", "INCOHERENT"),
    ]
    for row, _label, tag in picks:
        logger.info(
            "%s: R_bar=%.4f (ds=%s pos %d, crop %d, t=%d->%d)",
            tag,
            row["rbar"],
            row["dataset"],
            int(row["position"]),
            int(row["crop"]),
            int(row["t0"]),
            int(row["t1"]),
        )

    # Build per-(dataset, position) frame caches only for the picked rows
    # (skip if manual path already built them).
    if use_manual:
        pos_cache: dict[
            tuple[str, int], tuple[dict[int, np.ndarray], float]
        ] = pos_cache_simple
    else:
        pos_cache = {}
        for row, _, _ in picks:
            key = (row["dataset"], int(row["position"]))
            if key in pos_cache:
                continue
            cache, _crop_grid = _build_frame_cache(
                dataset_name=row["dataset"],
                position=int(row["position"]),
                channel=channel,
                level=level,
                n_frames=n_frames,
                annotations_to_exclude=annotations_to_exclude,
                df_dataset=df_dataset,
            )
            thresh = _compute_intensity_threshold(cache, intensity_pctl)
            pos_cache[key] = (cache, thresh)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        facecolor="white",
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.05, "hspace": 0.1},
    )
    for row_idx, (row, label, _tag) in enumerate(picks):
        cache, thresh = pos_cache[(row["dataset"], int(row["position"]))]
        _plot_pair(
            axes[row_idx],
            row,
            label,
            cache=cache,
            thresh=thresh,
            flow_scope=plot_scope,
            attachment=attachment,
        )

    save_dir = get_output_path("supp_fig_optical_flow")
    base_name = (
        f"optical_flow_panels_{dataset_name}_{channel}_"
        f"sel-{selection_scope}_plot-{plot_scope}"
    )
    for ext in (".png", ".svg"):
        save_plot_to_path(
            fig,
            save_dir,
            base_name,
            file_format=ext,
            dpi=300,
            show_and_close=False,
            tight_layout=False,
            bbox_inches="tight",
        )
    logger.info("Saved supplementary optical-flow figure to %s", save_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
