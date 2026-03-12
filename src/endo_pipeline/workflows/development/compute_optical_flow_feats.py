from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import TimepointAnnotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo-mode limits
# ---------------------------------------------------------------------------
_DEMO_MAX_DATASETS: int = 2
_DEMO_MAX_POSITIONS: int = 1
_DEMO_MAX_FRAMES: int = 5


def main(
    datasets: Datasets | None = None,
    positions: list[str] | None = None,
    channel: Sequence[str] = ("BF",),
    level: int | None = None,
    annotations_to_exclude: list[TimepointAnnotation] | None = None,
    max_dt: int | None = None,
    intensity_percentile: int | None = None,
    n_jobs: int = os.cpu_count() // 2,
    flow_scope: str = "image",
    upload: bool = True,
    visualize: bool = False,
    include_cell_piling: bool = False,
    include_pre_steady_state: bool = False,
) -> None:
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

    TVL1 attachment (lambda) is channel-aware:
        EGFP -> 7.5 (half the skimage default 15; wider normalised range).
        BF   -> 2.5 (z-score normalisation compresses dynamic range).

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
        Zarr resolution level.  None -> use DIFFAE_ZARR_RESOLUTION_LEVEL.
    annotations_to_exclude
        Explicit list of timepoint annotations to exclude.  When None
        (default), the list is built from include_cell_piling and
        include_pre_steady_state.  Passing an explicit list overrides
        both boolean flags.
    max_dt
        Maximum temporal gap (inclusive).  None -> DEFAULT_OPTICAL_FLOW_MAX_DT.
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
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import dask.array as da
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed  # type: ignore[import-untyped]
    from tqdm.auto import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_unannotated_timepoints_for_position,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_image
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.analyze.optical_flow import (
        build_crop_grid,
        build_optical_flow_feature_cols,
        compute_crop_flow,
        compute_image_pair_flow,
        default_annotations_to_exclude,
        pivot_flow_records,
        plot_demo_summary,
        resolve_attachment,
        resolve_percentile,
        save_and_upload,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_zarr_location_for_position,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL, DIMENSION_ORDER
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_FEATURE_COLUMN_NAMES,
        ColumnName,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_OMP_NUM_THREADS,
        DEFAULT_OPENBLAS_NUM_THREADS,
        DEFAULT_OPTICAL_FLOW_COLLECTION,
        DEFAULT_OPTICAL_FLOW_MAX_DT,
    )

    # Pin OpenMP to 1 thread per worker
    os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_OMP_NUM_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", DEFAULT_OPENBLAS_NUM_THREADS)

    # Resolve defaults that depend on settings imports
    if level is None:
        level = DIFFAE_ZARR_RESOLUTION_LEVEL
    if max_dt is None:
        max_dt = DEFAULT_OPTICAL_FLOW_MAX_DT

    columns_to_drop = list(DIFFAE_FEATURE_COLUMN_NAMES)

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
        annotations_to_exclude = default_annotations_to_exclude(
            include_cell_piling, include_pre_steady_state
        )
    assert flow_scope in ("image", "crop")

    channel = list(channel)
    pct = resolve_percentile(channel, intensity_percentile)
    attachment = resolve_attachment(channel)
    flow_cols = build_optical_flow_feature_cols(max_dt)
    is_bf = channel == ["BF"]

    logger.info(
        "Optical-flow extraction scope=%s dt=1..%d pct=%d attachment=%.1f channel=%s",
        flow_scope,
        max_dt,
        pct,
        attachment,
        channel,
    )

    # Shared manifests
    mm = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    dm_name = get_feature_dataframe_manifest_name(mm, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid")
    dm = load_dataframe_manifest(dm_name)

    all_parts: list[pd.DataFrame] = []

    for ds_i, ds_name in enumerate(datasets, 1):
        ds_t = time.time()
        logger.info("Dataset %d/%d: %s", ds_i, len(datasets), ds_name)

        ds_cfg = load_dataset_config(ds_name)
        df_ds = get_dataframe_for_dynamics_workflows(ds_name, dm, pca=None, filter_dataframe=True)
        pos_list = sorted(df_ds[ColumnName.POSITION].unique())
        if positions:
            pos_list = [p for p in pos_list if p in positions]
        if DEMO_MODE:
            pos_list = pos_list[:_DEMO_MAX_POSITIONS]
            logger.info("DEMO_MODE: limiting to position(s) %s", pos_list)

        ds_parts: list[pd.DataFrame] = []

        for pos_str in pos_list:
            pos_t = time.time()
            pos_idx = int(pos_str.lstrip("P"))
            valid_tp = get_unannotated_timepoints_for_position(
                ds_cfg, pos_idx, annotations_to_exclude
            )
            if DEMO_MODE:
                valid_tp = valid_tp[:_DEMO_MAX_FRAMES]
                logger.info("DEMO_MODE: limiting to %d frame(s)", len(valid_tp))
            valid_set = set(valid_tp)
            logger.info("%s valid_tp=%d/%d", pos_str, len(valid_tp), ds_cfg.duration)

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
            proj = da.log(img.std(axis=z_ax) + 1e-12) if is_bf else img.max(axis=z_ax)

            # Frame pairs
            pairs = [
                (t, t + d, d)
                for d in range(1, max_dt + 1)
                for t in valid_tp
                if (t + d) in valid_set
            ]
            needed = sorted({t for p in pairs for t in p[:2]})

            # Cache frames
            cache_t = time.time()
            needed_idxs = sorted(needed)
            needed_arr = proj[needed_idxs, 0].compute(
                scheduler="threads", num_workers=16
            )
            cache: dict[int, np.ndarray] = {}
            for j, t in enumerate(needed_idxs):
                frame = needed_arr[j].astype(np.float32, copy=False)
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
            del needed_arr
            logger.info("Cached %d frames in %.1fs", len(cache), time.time() - cache_t)

            # Intensity threshold
            thresh = (
                float(np.percentile(np.concatenate([f.ravel()[::10] for f in cache.values()]), pct))
                if pct > 0
                else -float("inf")
            )
            logger.info(
                "thresh(%d-pct)=%.6f pairs=%d crops=%d", pct, thresh, len(pairs), n_crops
            )

            # Compute flow
            records: list[dict] = []
            if flow_scope == "image":

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
                    _attachment=attachment,
                ):
                    f0, f1 = _cache[t0], _cache[t1]
                    return compute_image_pair_flow(
                        f0, f1, _sy, _ey, _sx, _ex, _cids, t0, dt, _thresh, _attachment,
                    )

                with ThreadPoolExecutor(max_workers=16) as pool:
                    futures = {
                        pool.submit(_image_pair, t0, t1, dt): (t0, dt) for t0, t1, dt in pairs
                    }
                    for fut in tqdm(
                        as_completed(futures), total=len(futures), desc=f"{ds_name} {pos_str}"
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
                        attachment,
                    )
                    for t0, t1, dt in pairs
                    for i in range(n_crops)
                ]
                records = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(compute_crop_flow)(*a)
                    for a in tqdm(args, desc=f"{ds_name} {pos_str}")
                )

            if visualize:
                plot_demo_summary(
                    cache, cg, ds_name, pos_str, thresh, results_dir, channel, flow_scope,
                    attachment,
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
            df_pos.drop(
                columns=[c for c in columns_to_drop if c in df_pos.columns], inplace=True
            )

            logger.info(
                "%s done %.1fs records=%d", pos_str, time.time() - pos_t, len(records)
            )
            ds_parts.append(df_pos)

        if ds_parts:
            df_ds_out = pd.concat(ds_parts, ignore_index=True)
            all_parts.append(df_ds_out)
            if upload:
                save_and_upload(ds_name, df_ds_out)
        logger.info("Dataset done %.1fs", time.time() - ds_t)

    df_out = pd.concat(all_parts, ignore_index=True)
    logger.info(
        "DONE %d rows x %d cols %.1fs", len(df_out), len(df_out.columns), time.time() - t0_all
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
