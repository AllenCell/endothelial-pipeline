import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.exposure import rescale_intensity
from skimage.registration import optical_flow_tvl1
from tqdm.auto import tqdm

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_datasets_in_collection,
    load_dataset_config,
)
from endo_pipeline.io import build_fms_annotations, get_output_path, load_image, upload_file_to_fms
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.diffae_features.optical_flow_utils import (
    build_crop_grid,
    build_optical_flow_feature_cols,
    compute_crop_flow,
    flow_stats,
    get_valid_timepoints,
    pivot_flow_records,
    resolve_percentile,
)
from endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    create_dataframe_manifest,
    get_feature_dataframe_manifest_name,
    get_zarr_location_for_position,
    load_dataframe_manifest,
    load_model_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL, DIMENSION_ORDER
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_FEATURE_COLUMN_NAMES, ColumnName
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_OMP_NUM_THREADS,
    DEFAULT_OPENBLAS_NUM_THREADS,
    DEFAULT_OPTICAL_FLOW_COLLECTION,
    DEFAULT_OPTICAL_FLOW_MANIFEST_NAME,
    DEFAULT_OPTICAL_FLOW_MAX_DT,
)

# Pin OpenMP to 1 thread per worker - joblib/ThreadPool handles parallelism
os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_OMP_NUM_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", DEFAULT_OPENBLAS_NUM_THREADS)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COLUMNS_TO_DROP = list(DIFFAE_FEATURE_COLUMN_NAMES)
_DEMO_MAX_DATASETS: int = 2
_DEMO_MAX_POSITIONS: int = 1
_DEMO_MAX_FRAMES: int = 5


def _default_annotations_to_exclude(
    include_cell_piling: bool,
    include_pre_steady_state: bool,
) -> list[TimepointAnnotation]:
    """Build the default exclusion list from convenience boolean flags."""
    excl: list[TimepointAnnotation] = []
    if not include_pre_steady_state:
        excl.append(TimepointAnnotation.NOT_STEADY_STATE)
    if not include_cell_piling:
        excl.append(TimepointAnnotation.CELL_PILING)
    return excl


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _plot_demo_summary(
    cache: dict[int, np.ndarray],
    crop_grid: pd.DataFrame,
    ds_name: str,
    pos_str: str,
    thresh: float,
    out_dir: Path,
) -> None:
    """Produce a quick diagnostic figure for one randomly chosen crop.

    Layout (1 x 4):
      (a) Red/Green composite of crop at t0 (red) and t1 (green).
      (b) Quiver plot of TVL1 flow (no background image).
      (c) Speed histogram (masked pixels only).
      (d) Angle histogram (masked pixels only).
    """
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    plt.style.use("endo_pipeline.figure")

    rng = np.random.default_rng(42)
    crop_row = crop_grid.iloc[rng.integers(len(crop_grid))]
    cidx = int(crop_row[ColumnName.CROP_INDEX])
    sx = int(crop_row[ColumnName.START_X])
    sy = int(crop_row[ColumnName.START_Y])
    ex = int(crop_row["end_x"])
    ey = int(crop_row["end_y"])
    cy, cx = ey - sy, ex - sx

    # Pick two consecutive cached frames
    sorted_tp = sorted(cache.keys())
    t0 = sorted_tp[rng.integers(max(1, len(sorted_tp) - 1))]
    t1_candidates = [t for t in sorted_tp if t > t0]
    t1 = t1_candidates[0] if t1_candidates else sorted_tp[-1]
    f0, f1 = cache[t0], cache[t1]

    crop0, crop1 = f0[sy:ey, sx:ex], f1[sy:ey, sx:ex]
    vf, uf = optical_flow_tvl1(crop0, crop1)
    speed = np.sqrt(uf**2 + vf**2)
    angle = np.arctan2(vf, uf)
    mask = (crop0 > thresh) | (crop1 > thresh)

    # Normalise crops to [0, 1] for the RGB composite
    def _norm(im: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(im, [2, 99.5])
        return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)

    c0n, c1n = _norm(crop0), _norm(crop1)

    # ---- Figure (1 x 4) ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor="white")

    # (a) Red/Green composite: t0 -> red, t1 -> green
    ax = axes[0]
    ax.set_facecolor("white")
    rgb = np.zeros((cy, cx, 3), dtype=np.float32)
    rgb[..., 0] = c0n  # red  = t0
    rgb[..., 1] = c1n  # green = t1
    ax.imshow(rgb, origin="upper")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="red", label=f"t={t0}"),
            Patch(facecolor="green", label=f"t={t1}"),
            Patch(facecolor="yellow", label="overlap"),
        ],
        fontsize=6,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        framealpha=0.7,
    )
    ax.set_title(f"(a) Composite  crop {cidx}")

    # (b) Quiver on blank white background
    ax = axes[1]
    ax.set_facecolor("white")
    step = max(1, cy // 8)
    Y, X = np.mgrid[0:cy:step, 0:cx:step]
    u_sub, v_sub = uf[::step, ::step], vf[::step, ::step]
    sp_sub = np.sqrt(u_sub**2 + v_sub**2)
    med_sp = float(np.median(sp_sub[sp_sub > 0])) if (sp_sub > 0).any() else 1.0
    q_scale = med_sp / (step * 0.6)
    qv = ax.quiver(
        X,
        Y,
        u_sub,
        v_sub,
        sp_sub,
        cmap="autumn",
        clim=[0, np.percentile(speed, 97)],
        angles="xy",
        scale_units="xy",
        scale=q_scale,
        width=0.008,
        headwidth=4,
        headlength=5,
        minshaft=1.5,
        alpha=0.85,
    )
    cbar = fig.colorbar(qv, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speed (px/frame)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_xlim(0, cx)
    ax.set_ylim(cy, 0)
    ax.set_aspect("equal")
    ax.set_title(f"(b) Quiver  t={t0}\u2192{t1}")

    # (c) Speed histogram
    ax = axes[2]
    ax.set_facecolor("white")
    if mask.any():
        sp_m = speed[mask]
        ax.hist(sp_m, bins=60, color="steelblue", edgecolor="white", linewidth=0.3, density=True)
        ax.axvline(
            sp_m.mean(),
            color="red",
            ls="--",
            lw=1.5,
            label=f"\u03bc = {sp_m.mean():.4f}",
        )
        ax.axvline(
            float(np.median(sp_m)),
            color="orange",
            ls="--",
            lw=1.5,
            label=f"median = {np.median(sp_m):.4f}",
        )
        ax.legend(fontsize=6)
    ax.set_xlabel("Speed (px/frame)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Speed distribution")

    # (d) Angle histogram
    ax = axes[3]
    ax.set_facecolor("white")
    if mask.any():
        ang_m = angle[mask]
        ax.hist(ang_m, bins=72, color="salmon", edgecolor="white", linewidth=0.3, density=True)
        cmean = float(np.arctan2(np.sin(ang_m).mean(), np.cos(ang_m).mean()))
        cstd = float(sp_stats.circstd(ang_m))
        ax.axvline(
            cmean,
            color="red",
            ls="--",
            lw=1.5,
            label=f"circ. mean = {cmean:.2f} rad",
        )
        ax.set_title(f"(d) \u03b8 distribution  \u03c3\u03b8 = {cstd:.4f} rad")
        ax.legend(fontsize=6)
    ax.set_xlabel("\u03b8 (rad)")
    ax.set_ylabel("Density")

    fig.suptitle(f"{ds_name} / {pos_str} / crop {cidx}  t={t0}->{t1}", fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / f"demo_quiver_{ds_name}_{pos_str}_crop{cidx}.png",
        dpi=150,
        facecolor="white",
    )
    logger.info("  Saved demo quiver figure to %s", out_dir)
    plt.show()


# ---------------------------------------------------------------------------
# FMS upload & manifest registration
# ---------------------------------------------------------------------------
def _save_parquet(dataset, df):
    out = get_output_path("optical_flow_manifest") / "results" / "optical_flow"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{dataset}_optical_flow_manifest.parquet"
    df.to_parquet(p, index=False)
    logger.info("  Saved parquet: %s", p)
    return p


def save_and_upload(dataset, df):
    """Save, upload to FMS, register in manifest. Returns FMS ID."""
    logger.info("Saving and uploading results for %s ...", dataset)
    p = _save_parquet(dataset, df)
    cfg = load_dataset_config(dataset)
    fms_id = upload_file_to_fms(p, build_fms_annotations(cfg), "parquet")
    logger.info("  Uploaded to FMS: %s", fms_id)
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


def main(
    datasets: list[str] | None = None,
    positions: list[str] | None = None,
    channel: Sequence[str] = ("BF",),
    level: int = DIFFAE_ZARR_RESOLUTION_LEVEL,
    annotations_to_exclude: list[TimepointAnnotation] | None = None,
    max_dt: int = DEFAULT_OPTICAL_FLOW_MAX_DT,
    intensity_percentile: int | None = None,
    n_jobs: int = os.cpu_count() // 2,
    flow_scope: str = "image",
    upload: bool = True,
    visualize: bool = False,
    include_cell_piling: bool = False,
    include_pre_steady_state: bool = False,
) -> pd.DataFrame:
    """Optical-flow feature extraction with multi-scale temporal coherence.

    Compute TVL1 optical flow between frame pairs at temporal gaps
    dt = 1, 2, ..., max_dt for every crop and timepoint.  Pixels whose
    intensity falls below a channel-aware percentile threshold are masked
    before computing per-crop summary statistics (mean/median speed,
    circular mean/std of angle, etc.).

    Flow scopes (flow_scope):
        "image" (default) - Run TVL1 once on the full-resolution image
            per frame pair, then slice flow vectors per crop.  Faster
            and avoids boundary artefacts.
        "crop" - Run TVL1 independently on each crop (legacy behaviour).

    Z-projection (channel-aware):
        BF   -> std(axis=Z) followed by log1p (enhances texture contrast).
        EGFP -> max-intensity projection.

    Intensity threshold (channel-aware):
        EGFP -> 95th percentile (sparse fluorescent signal; excludes background).
        BF   -> 0 (dense texture; all pixels contribute).

    Timepoint filtering:
        By default, timepoints annotated as cell-piling or pre-steady-state
        are excluded.  Use include_cell_piling / include_pre_steady_state
        to retain those frames, or pass an explicit annotations_to_exclude
        list for full control.

    Output:
        Results are saved as one parquet file per dataset under
        results/optical_flow/ and (optionally) uploaded to FMS.

    Parameters
    ----------
    datasets : list[str] | None
        Dataset names to process.  None -> all datasets in the default
        optical-flow collection.
    positions : list[str] | None
        Position identifiers to process (e.g. ["P0", "P1"]).
        None -> all positions in each dataset.
    channel : Sequence[str]
        Imaging channel(s) to load (e.g. ("EGFP",) or ("BF",)).
    level : int
        Zarr resolution level.
    annotations_to_exclude : list[TimepointAnnotation] | None
        Explicit list of timepoint annotations to exclude.  When None
        (default), the list is built from include_cell_piling and
        include_pre_steady_state.  Passing an explicit list overrides
        both boolean flags.
    max_dt : int
        Maximum temporal gap (inclusive).  Default 5.
    intensity_percentile : int | None
        Pixels below this percentile (computed across all cached frames)
        are masked out.  None -> auto-select based on channel
        (EGFP -> 95, BF -> 0).
    n_jobs : int
        Parallel workers used in "crop" flow scope (joblib/loky).
    flow_scope : str
        "image" or "crop" (see Flow scopes above).
    upload : bool
        If True, save parquet, upload to FMS, and register in the
        dataframe manifest.
    visualize : bool
        If True, produce diagnostic plots (R/G composite, quiver,
        speed & angle histograms) for one randomly chosen crop per
        (dataset, position) pair.  Saved to results/optical_flow/.
    include_cell_piling : bool
        If True, retain timepoints annotated as cell-piling.
        Default False (they are excluded).
    include_pre_steady_state : bool
        If True, retain timepoints before visual steady state.
        Default False (they are excluded).

    Returns
    -------
    pd.DataFrame
        Per-crop dataframe with {feature}_dt{1..max_dt} columns
        appended to the input metadata columns.
    """
    from endo_pipeline.cli import DEMO_MODE

    t0_all = time.time()
    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_OPTICAL_FLOW_COLLECTION)

    if DEMO_MODE:
        datasets = datasets[:_DEMO_MAX_DATASETS]
        upload = False
        visualize = True
        logger.info(
            "DEMO_MODE is ON. Processing %d dataset(s), %d position(s) each, upload disabled.",
            _DEMO_MAX_DATASETS,
            _DEMO_MAX_POSITIONS,
        )

    results_dir = get_output_path("optical_flow_manifest") / "results" / "optical_flow"
    results_dir.mkdir(parents=True, exist_ok=True)
    if annotations_to_exclude is None:
        annotations_to_exclude = _default_annotations_to_exclude(
            include_cell_piling, include_pre_steady_state
        )
    assert flow_scope in ("image", "crop")

    channel = list(channel)
    pct = resolve_percentile(channel, intensity_percentile)
    flow_cols = build_optical_flow_feature_cols(max_dt)
    is_bf = channel == ["BF"]

    logger.info(
        "Optical-flow extraction  scope=%s  dt=1..%d  pct=%d  channel=%s",
        flow_scope,
        max_dt,
        pct,
        channel,
    )

    # Shared manifests
    mm = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    dm_name = get_feature_dataframe_manifest_name(mm, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid")
    dm = load_dataframe_manifest(dm_name)

    all_parts: list[pd.DataFrame] = []

    for ds_i, ds_name in enumerate(datasets, 1):
        ds_t = time.time()
        logger.info("=" * 50)
        logger.info("Dataset %d/%d: %s", ds_i, len(datasets), ds_name)
        logger.info("=" * 50)

        ds_cfg = load_dataset_config(ds_name)
        df_ds = get_dataframe_for_dynamics_workflows(ds_name, dm, pca=None, filter_dataframe=True)
        pos_list = sorted(df_ds[ColumnName.POSITION].unique())
        if positions:
            pos_list = [p for p in pos_list if p in positions]
        if DEMO_MODE:
            pos_list = pos_list[:_DEMO_MAX_POSITIONS]
            logger.info("  DEMO_MODE: limiting to position(s) %s", pos_list)

        ds_parts: list[pd.DataFrame] = []

        for pos_str in pos_list:
            pos_t = time.time()
            pos_idx = int(pos_str.lstrip("P"))
            valid_tp = get_valid_timepoints(ds_cfg, pos_idx, annotations_to_exclude)
            if DEMO_MODE:
                valid_tp = valid_tp[:_DEMO_MAX_FRAMES]
                logger.info("  DEMO_MODE: limiting to %d frame(s)", len(valid_tp))
            valid_set = set(valid_tp)
            logger.info("  %s  valid_tp=%d/%d", pos_str, len(valid_tp), ds_cfg.duration)

            df_pos = df_ds[
                (df_ds[ColumnName.POSITION] == pos_str)
                & (df_ds[ColumnName.TIMEPOINT].isin(valid_tp))
            ].copy()
            if df_pos.empty:
                continue
            df_pos["dataset"] = ds_name

            # Crop grid
            cg = build_crop_grid(df_pos)
            sx, sy = cg[ColumnName.START_X].values.astype(int), cg[
                ColumnName.START_Y
            ].values.astype(int)
            ex, ey = cg["end_x"].values.astype(int), cg["end_y"].values.astype(int)
            cids = cg[ColumnName.CROP_INDEX].values
            n_crops = len(cg)

            # Z-project
            zarr_loc = get_zarr_location_for_position(ds_cfg, pos_idx)
            img = load_image(zarr_loc, channels=channel, level=level, compute=False)
            z_ax = DIMENSION_ORDER.index("Z")
            proj = da.log1p(img.std(axis=z_ax)) if is_bf else img.max(axis=z_ax)

            # Frame pairs
            pairs = [
                (t, t + d, d)
                for d in range(1, max_dt + 1)
                for t in valid_tp
                if (t + d) in valid_set
            ]
            needed = sorted({t for p in pairs for t in p[:2]})

            # Cache frames - fancy-index lets dask parallelise NFS reads per T
            cache_t = time.time()
            needed_idxs = sorted(needed)
            needed_arr = proj[needed_idxs, 0].compute(
                scheduler="threads", num_workers=16
            )  # (n_needed, y, x)
            cache: dict[int, np.ndarray] = {}
            for j, t in enumerate(needed_idxs):
                cache[t] = rescale_intensity(
                    needed_arr[j].astype(np.float32, copy=False), out_range=(0.0, 1.0)
                )
            del needed_arr
            logger.info("    cached %d frames in %.1fs", len(cache), time.time() - cache_t)

            # Intensity threshold (subsample 10% of pixels for speed + memory)
            thresh = (
                float(np.percentile(np.concatenate([f.ravel()[::10] for f in cache.values()]), pct))
                if pct > 0
                else 0.0
            )
            logger.info(
                "    thresh(%d-pct)=%.6f  pairs=%d  crops=%d", pct, thresh, len(pairs), n_crops
            )

            # Compute flow (threaded - TVL1 is C/GIL-free, OMP pinned to 1)
            records: list[dict] = []
            if flow_scope == "image":

                # Captured variables (cache, sy, ey, sx, ex, cids, thresh,
                # n_crops) are stable for the lifetime of this executor block.
                def _image_pair(t0, t1, dt):  # noqa: B023
                    f0, f1 = cache[t0], cache[t1]  # noqa: B023
                    vf, uf = optical_flow_tvl1(f0, f1)
                    return [
                        flow_stats(
                            uf[sy[i] : ey[i], sx[i] : ex[i]],  # noqa: B023
                            vf[sy[i] : ey[i], sx[i] : ex[i]],  # noqa: B023
                            f0[sy[i] : ey[i], sx[i] : ex[i]],  # noqa: B023
                            f1[sy[i] : ey[i], sx[i] : ex[i]],  # noqa: B023
                            int(cids[i]),  # noqa: B023
                            t0,
                            dt,
                            thresh,  # noqa: B023
                        )
                        for i in range(n_crops)  # noqa: B023
                    ]

                with ThreadPoolExecutor(max_workers=16) as pool:
                    futures = {
                        pool.submit(_image_pair, t0, t1, dt): (t0, dt) for t0, t1, dt in pairs
                    }
                    for fut in tqdm(
                        as_completed(futures), total=len(futures), desc=f"  {ds_name} {pos_str}"
                    ):
                        records.extend(fut.result())
            else:
                args = [
                    (
                        cache[t0][sy[i] : ey[i], sx[i] : ex[i]].copy(),
                        cache[t1][sy[i] : ey[i], sx[i] : ex[i]].copy(),
                        cids[i],
                        t0,
                        dt,
                        thresh,
                    )
                    for t0, t1, dt in pairs
                    for i in range(n_crops)
                ]
                records = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(compute_crop_flow)(*a)
                    for a in tqdm(args, desc=f"  {ds_name} {pos_str}")
                )
            # Visualization (before clearing cache)
            if visualize:
                _plot_demo_summary(
                    cache,
                    cg,
                    ds_name,
                    pos_str,
                    thresh,
                    results_dir,
                )

            cache.clear()

            # Pivot & merge
            df_pivot = pivot_flow_records(records, max_dt)
            df_pos = df_pos.merge(
                df_pivot,
                left_on=[ColumnName.CROP_INDEX, ColumnName.TIMEPOINT],
                right_on=["crop_index", "timepoint"],
                how="left",
            ).drop(columns=["crop_index", "timepoint"], errors="ignore")

            for c in flow_cols:
                if c not in df_pos.columns:
                    df_pos[c] = np.nan
            df_pos.drop(columns=[c for c in _COLUMNS_TO_DROP if c in df_pos.columns], inplace=True)

            logger.info(
                "    %s done  %.1fs  records=%d", pos_str, time.time() - pos_t, len(records)
            )
            ds_parts.append(df_pos)

        if ds_parts:
            df_ds_out = pd.concat(ds_parts, ignore_index=True)
            all_parts.append(df_ds_out)
            if upload:
                save_and_upload(ds_name, df_ds_out)
        logger.info("  dataset done %.1fs", time.time() - ds_t)

    df_out = pd.concat(all_parts, ignore_index=True)
    logger.info(
        "DONE  %d rows x %d cols  %.1fs", len(df_out), len(df_out.columns), time.time() - t0_all
    )
    return df_out


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
