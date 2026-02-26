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

# Pin OpenMP to 1 thread per worker — joblib/ThreadPool handles parallelism
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
    DEFAULT_OPTICAL_FLOW_COLLECTION,
    DEFAULT_OPTICAL_FLOW_MANIFEST_NAME,
    DEFAULT_OPTICAL_FLOW_MAX_DT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COLUMNS_TO_DROP = list(DIFFAE_FEATURE_COLUMN_NAMES)


# ---------------------------------------------------------------------------
# FMS upload & manifest registration
# ---------------------------------------------------------------------------
def _save_parquet(dataset, df):
    out = get_output_path("optical_flow_manifest")
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
) -> pd.DataFrame:
    """Optical-flow feature extraction - multi-scale angle coherence.

    For each crop and timepoint, TVL1 optical flow is computed at frame gaps
    dt = 1, 2, …, ``max_dt``.  Pixels below the intensity-percentile
    threshold are excluded before computing summary statistics.

    Flow scopes (``flow_scope``):
      ``"image"``: TVL1 on the full image per frame pair, slice per crop.
      ``"crop"``:  TVL1 independently per crop (legacy).

    Z-projection is channel-aware:
      BF  → std(axis=Z) + log1p   (texture contrast)
      FL  → max-intensity projection

    Intensity threshold is channel-aware:
      EGFP → 95th percentile (sparse signal, exclude background)
      BF   → 0 (dense texture, use all pixels)

    Results saved as parquet per dataset and uploaded to FMS.

    Parameters
    ----------
    datasets, positions
        Dataset/position filters.  *None* → use all available.
    channel
        Channel to load (e.g. ``("EGFP",)`` or ``("BF",)``).
    max_dt
        Maximum frame gap (inclusive).  Default 5.
    intensity_percentile
        Pixels below this percentile (across all cached frames) are masked.
        *None* → auto-select based on channel (EGFP=95, BF=0).
    flow_scope
        ``"image"`` runs TVL1 once on the whole image then slices per crop.
        ``"crop"`` runs TVL1 independently per crop (legacy).
    n_jobs
        Parallel workers for crop-level flow scope.
    upload
        Whether to save parquet, upload to FMS, and register manifest.

    Returns
    -------
        Per-crop dataframe with ``{feature}_dt{1..max_dt}`` columns.
    """
    t0_all = time.time()
    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_OPTICAL_FLOW_COLLECTION)
    if annotations_to_exclude is None:
        annotations_to_exclude = [
            TimepointAnnotation.NOT_STEADY_STATE,
            TimepointAnnotation.CELL_PILING,
        ]
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

        ds_parts: list[pd.DataFrame] = []

        for pos_str in pos_list:
            pos_t = time.time()
            pos_idx = int(pos_str.lstrip("P"))
            valid_tp = get_valid_timepoints(ds_cfg, pos_idx, annotations_to_exclude)
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

            # Cache frames — fancy-index lets dask parallelise NFS reads per T
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

            # Compute flow (threaded — TVL1 is C/GIL-free, OMP pinned to 1)
            records: list[dict] = []
            if flow_scope == "image":

                def _image_pair(
                    t0,
                    t1,
                    dt,
                    cache=cache,
                    sy=sy,
                    ey=ey,
                    sx=sx,
                    ex=ex,
                    cids=cids,
                    thresh=thresh,
                    n_crops=n_crops,
                ):
                    f0, f1 = cache[t0], cache[t1]
                    vf, uf = optical_flow_tvl1(f0, f1)
                    return [
                        flow_stats(
                            uf[sy[i] : ey[i], sx[i] : ex[i]],
                            vf[sy[i] : ey[i], sx[i] : ex[i]],
                            f0[sy[i] : ey[i], sx[i] : ex[i]],
                            f1[sy[i] : ey[i], sx[i] : ex[i]],
                            int(cids[i]),
                            t0,
                            dt,
                            thresh,
                        )
                        for i in range(n_crops)
                    ]

                with ThreadPoolExecutor(max_workers=32) as pool:
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
