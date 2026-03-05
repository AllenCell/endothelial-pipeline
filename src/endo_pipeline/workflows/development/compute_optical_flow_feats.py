import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from joblib import Parallel, delayed  # type: ignore[import-untyped]
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
    compute_image_pair_flow,
    compute_tvl1,
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
_DEMO_SCAN_N_CROPS: int = 6
_DEMO_SCAN_N_PAIRS: int = 10
_QUIVER_GRID_DIVISIONS: int = 8


def _default_annotations_to_exclude(
    include_cell_piling: bool = False,
    include_pre_steady_state: bool = False,
) -> list[TimepointAnnotation]:
    """Build the default timepoint-annotation exclusion list.

    Quality annotations (scope errors, temporary artifacts, XY/Z shifts,
    unfed) are **always** excluded — computing optical flow against
    corrupted frames would contaminate neighbouring good frames.  This
    keeps :func:`get_valid_timepoints` in sync with the dataframe
    filter applied by :func:`get_dataframe_for_dynamics_workflows`.

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
    # Quality annotations — always excluded to prevent computing flow
    # against corrupted frames (and contaminating adjacent good frames).
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
    # Lifecycle annotations — toggleable via flags.
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
    channel: list[str],
    flow_scope: str,
) -> None:
    """Produce a coherent-vs-incoherent diagnostic figure (2 rows x 4 cols).

    Scans a subsample of (crop, timepoint) pairs, picks the pair with
    lowest circular-std (most coherent) and highest (most incoherent),
    then plots for each:
      (a) Red/Green composite of crop at t0 (red) and t1 (green).
      (b) Quiver plot of TVL1 flow.
      (c) Speed histogram (masked pixels only, with +/-1-sigma band).
      (d) Angle histogram (masked pixels only).

    Skips plotting entirely if the cache contains fewer than 2 frames
    or the subsample scan yields fewer than 2 valid records.

    Parameters
    ----------
    cache
        Mapping from timepoint index to its 2-D intensity frame,
        shape ``(H, W)``, as produced by the caching step in
        :func:`main`.
    crop_grid
        One row per spatial crop with columns for crop index and
        start/end x/y coordinates (see :func:`build_crop_grid`).
    ds_name
        Dataset name, used in the figure title and output filename.
    pos_str
        Position identifier (e.g. ``"P0"``), used in the figure
        title and output filename.
    thresh
        Intensity threshold for foreground masking.  Pixels where
        both frames are at or below this value are excluded from
        flow statistics.
    out_dir
        Directory where the PNG figure is saved.  Created if it
        does not exist.
    channel
        Imaging channel name(s) (e.g. ``["BF"]``), shown in the
        figure suptitle and encoded in the output filename.
    flow_scope
        Flow computation strategy (``"image"`` or ``"crop"``),
        included in the figure title and output filename.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy import stats as sp_stats

    plt.style.use("endo_pipeline.figure")

    # ------------------------------------------------------------------------
    # Scan a subsample of crops x timepoints to find coherent/incoherent flow
    # ------------------------------------------------------------------------
    sorted_tp = sorted(cache.keys())
    if len(sorted_tp) < 2:
        logger.warning("  Only %d cached frame(s) — skipping demo plot", len(sorted_tp))
        return

    cids = crop_grid[ColumnName.CROP_INDEX].values
    sx_arr = crop_grid[ColumnName.START_X].values.astype(int)
    sy_arr = crop_grid[ColumnName.START_Y].values.astype(int)
    ex_arr = crop_grid["end_x"].values.astype(int)
    ey_arr = crop_grid["end_y"].values.astype(int)

    # Subsampling
    crop_step = max(1, len(cids) // _DEMO_SCAN_N_CROPS)
    scan_cids = range(0, len(cids), crop_step)

    # Build consecutive pairs from sorted cache keys
    all_pairs = [(sorted_tp[i], sorted_tp[i + 1]) for i in range(len(sorted_tp) - 1)]
    pair_step = max(1, len(all_pairs) // _DEMO_SCAN_N_PAIRS)
    scan_pairs = all_pairs[::pair_step]

    records = []
    # Cache full-image flow fields when scope is "image" to avoid redundant
    # TVL1 calls across crops for the same frame pair.
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
                    _image_flow_cache[(t0, t1)] = compute_tvl1(f0, f1)
                uf_full, vf_full = _image_flow_cache[(t0, t1)]
                uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]
            else:
                uf, vf = compute_tvl1(c0, c1)
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
        logger.warning("  Scan produced <2 valid records — skipping demo plot")
        return

    best = scan_df.loc[scan_df["circ_std"].idxmin()]
    worst = scan_df.loc[scan_df["circ_std"].idxmax()]

    logger.info(
        "  Demo scan: %d pairs → coherent σθ=%.4f (crop %d, t=%d→%d)  "
        "incoherent σθ=%.4f (crop %d, t=%d→%d)",
        len(scan_df),
        best["circ_std"],
        int(best["crop"]),
        int(best["t0"]),
        int(best["t1"]),
        worst["circ_std"],
        int(worst["crop"]),
        int(worst["t0"]),
        int(worst["t1"]),
    )

    # ------------------------------------------------------------------
    # Helper: plot one row (4 panels)
    # ------------------------------------------------------------------
    def _plot_row(axes, row, label):
        t0, t1 = int(row["t0"]), int(row["t1"])
        _sx, _sy = int(row["sx"]), int(row["sy"])
        _ex, _ey = int(row["ex"]), int(row["ey"])
        _cidx = int(row["crop"])
        cy, cx = _ey - _sy, _ex - _sx

        f0, f1 = cache[t0], cache[t1]
        c0, c1 = f0[_sy:_ey, _sx:_ex], f1[_sy:_ey, _sx:_ex]
        if flow_scope == "image":
            uf_full, vf_full = compute_tvl1(f0, f1)
            uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]
        else:
            uf, vf = compute_tvl1(c0, c1)
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

        # (c) Speed histogram
        ax = axes[2]
        ax.set_facecolor("white")
        if mask.any():
            sp_m = sp[mask]
            ax.hist(
                sp_m, bins=60, color="steelblue", edgecolor="white", linewidth=0.3, density=True
            )
            ax.axvline(
                sp_m.mean(), color="red", ls="--", lw=1.5, label=f"\u03bc = {sp_m.mean():.4f}"
            )
            ax.axvline(
                float(np.median(sp_m)),
                color="orange",
                ls="--",
                lw=1.5,
                label=f"median = {np.median(sp_m):.4f}",
            )
            ax.axvspan(
                sp_m.mean() - sp_m.std(),
                sp_m.mean() + sp_m.std(),
                color="red",
                alpha=0.08,
                label=f"\u00b11\u03c3  (\u03c3={sp_m.std():.4f})",
            )
            ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("Speed (px/frame)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title("(c) Speed distribution", fontsize=9)
        ax.tick_params(labelsize=7)

        # (d) Angle histogram
        ax = axes[3]
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
            f"(d) $\\theta$ distribution  $\\sigma_{{\\theta}}$ = {cstd:.4f} rad", fontsize=9
        )
        ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # Build the 2x4 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), facecolor="white")

    _plot_row(axes[0], best, r"COHERENT (low $\sigma_{\theta}$)")
    _plot_row(axes[1], worst, r"INCOHERENT (high $\sigma_{\theta}$)")

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
    logger.info("  Saved coherent-vs-incoherent figure to %s", out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FMS upload & manifest registration
# ---------------------------------------------------------------------------
def _save_parquet(dataset: str, df: pd.DataFrame) -> Path:
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
    out = get_output_path("optical_flow") / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{dataset}_optical_flow_manifest.parquet"
    df.to_parquet(p, index=False)
    logger.info("  Saved parquet: %s", p)
    return p


def save_and_upload(dataset: str, df: pd.DataFrame) -> str:
    """Save a parquet, upload it to FMS, and register it in the manifest.

    Persists *df* via :func:`_save_parquet`, uploads the file to FMS
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
        "crop" - Run TVL1 independently on each crop.

    Z-projection & normalization (channel-aware, matches DiffAE training):
        BF   -> std(axis=Z) -> log(x+1e-12) -> clip [0.1, 99.9] pctl -> z-score.
        EGFP -> max(axis=Z) -> clip [10, 98] pctl -> scale to [-1, 1].

    TVL1 attachment (lambda) is set to 7.5 (half the default 15) to
    compensate for the wider normalised intensity range.

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
    datasets
        Dataset names to process.  None -> all datasets in the default
        optical-flow collection.
    positions
        Position identifiers to process (e.g. ["P0", "P1"]).
        None -> all positions in each dataset.
    channel
        Imaging channel(s) to load (e.g. ("EGFP",) or ("BF",)).
    level
        Zarr resolution level.
    annotations_to_exclude
        Explicit list of timepoint annotations to exclude.  When None
        (default), the list is built from include_cell_piling and
        include_pre_steady_state.  Passing an explicit list overrides
        both boolean flags.
    max_dt
        Maximum temporal gap (inclusive).  Default 5.
    intensity_percentile
        Pixels below this percentile (computed across all cached frames)
        are masked out.  None -> auto-select based on channel
        (EGFP -> 95, BF -> 0).
    n_jobs
        Parallel workers used in "crop" flow scope (joblib/loky).
    flow_scope
        "image" or "crop" (see Flow scopes above).
    upload
        If True, save parquet, upload to FMS, and register in the
        dataframe manifest.
    visualize
        If True, produce diagnostic plots (R/G composite, quiver,
        speed & angle histograms) for one randomly chosen crop per
        (dataset, position) pair.  Saved to results/optical_flow/.
    include_cell_piling
        If True, retain timepoints annotated as cell-piling.
        Default False (they are excluded).
    include_pre_steady_state
        If True, retain timepoints before visual steady state.
        Default False (they are excluded).

    Returns
    -------
        Per-crop dataframe with {feature}_dt{1..max_dt} columns
        appended to the input metadata columns.
    """
    from endo_pipeline.cli import DEMO_MODE

    # To measure the time it takes for the computations!
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

    results_dir = get_output_path("optical_flow") / "plots"
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
            # BF: std Z-projection -> log(x + 1e-12) to match DiffAE training
            proj = da.log(img.std(axis=z_ax) + 1e-12) if is_bf else img.max(axis=z_ax)

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
                frame = needed_arr[j].astype(np.float32, copy=False)
                if is_bf:
                    # Match DiffAE BF pipeline:
                    #   Clipd [0.1, 99.9] -> NormalizeIntensityd (z-score)
                    lo, hi = np.percentile(frame, [0.1, 99.9])
                    frame = np.clip(frame, lo, hi)
                    std = frame.std()
                    frame = (frame - frame.mean()) / (std if std > 0 else 1.0)
                else:
                    # Match DiffAE EGFP pipeline:
                    #   ScaleIntensityRangePercentilesd(lower=10, upper=98,
                    #       b_min=-1, b_max=1, clip=True)
                    lo, hi = np.percentile(frame, [10, 98])
                    frame = np.clip(frame, lo, hi)
                    frame = (frame - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
                cache[t] = frame
            del needed_arr
            logger.info("    cached %d frames in %.1fs", len(cache), time.time() - cache_t)

            # Intensity threshold (subsample 10% of pixels for speed + memory)
            # pct=0 (BF) → -inf so all pixels pass regardless of normalization
            thresh = (
                float(np.percentile(np.concatenate([f.ravel()[::10] for f in cache.values()]), pct))
                if pct > 0
                else -float("inf")
            )
            logger.info(
                "    thresh(%d-pct)=%.6f  pairs=%d  crops=%d", pct, thresh, len(pairs), n_crops
            )

            # Compute flow (threaded - TVL1 is C/GIL-free, OMP pinned to 1)
            records: list[dict] = []
            if flow_scope == "image":

                # Bind loop-stable arrays as defaults so ruff B023 is satisfied.
                def _image_pair(
                    t0,
                    t1,
                    dt,
                    _cache=cache,
                    _sy=sy,
                    _ey=ey,
                    _sx=sx,
                    _ex=ex,
                    _cids=cids,
                    _thresh=thresh,
                ):
                    f0, f1 = _cache[t0], _cache[t1]
                    return compute_image_pair_flow(
                        f0,
                        f1,
                        _sy,
                        _ey,
                        _sx,
                        _ex,
                        _cids,
                        t0,
                        dt,
                        _thresh,
                    )

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
                    channel,
                    flow_scope,
                )

            cache.clear()

            # Pivot & merge
            df_pivot = pivot_flow_records(records)
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
