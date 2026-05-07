import logging
from typing import Literal

from endo_pipeline.cli import CropPattern, Datasets, FloatList
from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL
from endo_pipeline.settings.optical_flow import (
    DEFAULT_EMA_ALPHAS,
    DEFAULT_OPTICAL_FLOW_MAX_DT,
    DEFAULT_SPEED_THRESHOLD,
    NUM_IO_WORKERS,
)

logger = logging.getLogger(__name__)


def main(  # noqa: C901
    datasets: Datasets | None = None,
    positions: list[int] | None = None,
    channel: Literal["BF", "EGFP"] = "BF",
    level: int = DIFFAE_ZARR_RESOLUTION_LEVEL,
    annotations_to_exclude: list[TimepointAnnotation] | None = None,
    max_dt: int = DEFAULT_OPTICAL_FLOW_MAX_DT,
    intensity_percentile: int | None = None,
    n_jobs: int | None = None,
    n_io_workers: int = NUM_IO_WORKERS,
    flow_scope: Literal["image", "crop"] = "image",
    crop_pattern: CropPattern = "grid",
    upload_to_fms: bool = False,
    visualize_optical_flow: bool = False,
    include_cell_piling: bool = False,
    include_pre_steady_state: bool = False,
    include_all_conditions: bool = False,
    compute_block_coherence: bool = False,
    compute_fast_coherence: bool = False,
    compute_radial_coherence: bool = False,
    ema_alphas: FloatList = list(DEFAULT_EMA_ALPHAS),
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
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

    Crop patterns (crop_pattern):
        "grid" (default) - Use the fixed-grid crop layout from the DiffAE
            feature dataframe.
        "tracked" - Use per-frame tracked crop coordinates from the DiffAE
            feature dataframe (requires ``TRACK_ID``, ``START_X``,
            ``START_Y``, ``CROP_SIZE_X`` columns).

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
        Position indices to process (e.g. ``[0, 1]``).
        ``None`` processes all positions in each dataset.
    channel
        Imaging channel to load (``"BF"`` or ``"EGFP"``).
    level
        Zarr resolution level.
    annotations_to_exclude
        Explicit list of timepoint annotations to exclude.  When ``None``
        (default), the list is built from *include_cell_piling* and
        *include_pre_steady_state*.  Pass an empty list ``[]`` to
        bypass **all** annotation filtering (including quality
        annotations).  Passing a non-empty list overrides both boolean
        flags.
    max_dt
        Maximum temporal gap (inclusive).
    intensity_percentile
        Pixels below this percentile (computed across all cached frames)
        are masked out.  None -> auto-select based on channel
        (EGFP -> 95, BF -> 0).
    n_jobs
        Parallel workers used in "crop" flow scope (joblib/loky).
    n_io_workers
        Concurrent I/O workers for dask frame loading and
        ``ThreadPoolExecutor`` in image-scope flow.
    flow_scope
        "image" or "crop" (see Flow scopes above).
    crop_pattern
        "grid" or "tracked" (see Crop patterns above).
    upload_to_fms
        If True, save parquet, upload to FMS, and register in the
        dataframe manifest.
    visualize_optical_flow
        If True, produce diagnostic plots (R/G composite, quiver,
        speed & angle histograms) for one randomly chosen crop per
        (dataset, position) pair.  Saved to results/optical_flow/.
    include_cell_piling
        If True, retain timepoints annotated as cell-piling.
    include_pre_steady_state
        If True, retain timepoints before visual steady state.
    include_all_conditions
        If True, bypass **all** annotation filtering — including quality
        annotations (scope errors, temp artifacts, XY/Z shifts, unfed).
        Overrides both *include_cell_piling* and *include_pre_steady_state*.
    compute_block_coherence
        If True, compute multi-scale block-averaged coherence statistics
        (``optical_flow_angle_std_box{N}``) for each box size.
    compute_fast_coherence
        If True, compute coherence metrics over pixels whose speed
        exceeds *speed_threshold*.
    compute_radial_coherence
        If True, compute radial coherence metrics (dot product of
        unit flow with unit radial vector from crop centre).
    ema_alphas
        EMA smoothing alpha values for temporal coherence smoothing.
    speed_threshold
        Minimum pixel speed for the "fast" coherence features.
        Only used when *compute_fast_coherence* is True.
    """
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from functools import partial

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
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_dataframe,
        load_image,
        make_name_unique,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.optical_flow import (
        build_crop_grid,
        build_ema_stems,
        build_optical_flow_feature_cols,
        compute_crop_flow,
        compute_image_pair_flow,
        default_annotations_to_exclude,
        pivot_flow_records,
        plot_demo_summary,
        plot_tracked_crop_coherence_timeseries,
        resolve_attachment,
        resolve_percentile,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        get_zarr_location_for_position,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings import DIMENSION_ORDER
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.optical_flow import (
        DEFAULT_OMP_NUM_THREADS,
        DEFAULT_OPENBLAS_NUM_THREADS,
        DEFAULT_OPTICAL_FLOW_COLLECTION,
        DEMO_MAX_DATASETS,
        DEMO_MAX_FRAMES,
        DEMO_MAX_POSITIONS,
        DEMO_MAX_TRACKED_CROPS_TO_PLOT,
        OPTICAL_FLOW_COLUMNS_TO_COMPUTE,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    # Pin OpenMP to 1 thread per worker
    os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_OMP_NUM_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", DEFAULT_OPENBLAS_NUM_THREADS)

    is_tracked = crop_pattern == "tracked"

    # Tracked crops require image-scope TVL1 — reject early.
    if is_tracked and flow_scope == "crop":
        raise ValueError(
            "crop_pattern='tracked' is not supported with flow_scope='crop'. "
            "Use flow_scope='image' for tracked crops."
        )

    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_OPTICAL_FLOW_COLLECTION)

    if DEMO_MODE:
        datasets = datasets[:DEMO_MAX_DATASETS]
        upload_to_fms = False
        visualize_optical_flow = True
        logger.info(
            "DEMO_MODE is ON (tracked=%s). Processing %d dataset(s), %d position(s) each, "
            "upload disabled.",
            is_tracked,
            DEMO_MAX_DATASETS,
            DEMO_MAX_POSITIONS,
        )

    results_dir = get_output_path("optical_flow", "plots")
    if annotations_to_exclude is None:
        if include_all_conditions:
            annotations_to_exclude = []
        else:
            annotations_to_exclude = default_annotations_to_exclude(
                include_cell_piling, include_pre_steady_state
            )

    logger.info(
        "annotations_to_exclude (%d): %s",
        len(annotations_to_exclude),
        (
            [a.value for a in annotations_to_exclude]
            if annotations_to_exclude
            else "[] (no filtering)"
        ),
    )
    intensity_pctl = resolve_percentile(channel, intensity_percentile)
    attachment = resolve_attachment(channel)
    flow_columns = build_optical_flow_feature_cols(
        max_dt,
        compute_block_coherence=compute_block_coherence,
        compute_fast_coherence=compute_fast_coherence,
        compute_radial_coherence=compute_radial_coherence,
        ema_alphas=ema_alphas,
    )
    is_bf = channel == "BF"

    logger.info(
        "Optical-flow extraction --> scope=%s | crop_pattern=%s | dt=1..%d "
        "| percentile=%d | attachment=%.1f | channel=%s "
        "| block_coherence=%s | fast_coherence=%s | radial_coherence=%s "
        "| ema_alphas=%s | speed_threshold=%.2f",
        flow_scope,
        crop_pattern,
        max_dt,
        intensity_pctl,
        attachment,
        channel,
        compute_block_coherence,
        compute_fast_coherence,
        compute_radial_coherence,
        ema_alphas,
        speed_threshold,
    )

    # Load dataframe with diffae feature metadata (no filtering yet) to get crop
    # coordinates and timepoints for each dataset/position.
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # For tracked crops, we also need TRACK_ID, CROP_SIZE_X
    columns_to_compute = list(OPTICAL_FLOW_COLUMNS_TO_COMPUTE[crop_pattern])

    # Load or create optical flow manifest and set parameters before the loop so
    # that it is available even if the workflow fails partway through. Add
    # suffix to manifest name in demo mode to avoid overwriting real results
    # with partial demo results.
    demo_suffix = "_demo" if DEMO_MODE else ""
    optical_flow_manifest = create_dataframe_manifest(
        f"optical_flow_bf_{crop_pattern}{demo_suffix}",
        workflow_name=__file__,
    )
    optical_flow_manifest.parameters = {
        "channel": channel,
        "level": level,
        "max_dt": max_dt,
        "flow_scope": flow_scope,
        "crop_pattern": crop_pattern,
        "intensity_percentile": intensity_pctl,
        "attachment": attachment,
        "compute_block_coherence": compute_block_coherence,
        "compute_fast_coherence": compute_fast_coherence,
        "compute_radial_coherence": compute_radial_coherence,
        "ema_alphas": list(ema_alphas),
        "speed_threshold": speed_threshold,
        "annotations_excluded": [a.value for a in annotations_to_exclude],
    }
    save_dataframe_manifest(optical_flow_manifest)

    # set output directory for dataframes
    output_dir = get_output_path("optical_flow", "dataframes")

    for dataset_idx, dataset_name in enumerate(datasets, 1):
        dataset_start = time.time()
        logger.info("Dataset %d/%d: %s", dataset_idx, len(datasets), dataset_name)

        dataset_config = load_dataset_config(dataset_name)
        df_dataset = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df_dataset_: pd.DataFrame = df_dataset[columns_to_compute].compute()

        position_list = sorted(df_dataset_[Column.POSITION].unique().tolist())
        if positions:
            position_list = [p for p in position_list if p in positions]
        if DEMO_MODE:
            position_list = position_list[:DEMO_MAX_POSITIONS]
            logger.info("DEMO_MODE: limiting to position(s) %s", position_list)

        dataset_parts: list[pd.DataFrame] = []

        for position in position_list:
            position_start = time.time()
            valid_timepoints = get_unannotated_timepoints_for_position(
                dataset_config, position, annotations_to_exclude
            )
            if DEMO_MODE:
                valid_timepoints = valid_timepoints[:DEMO_MAX_FRAMES]
                logger.info("DEMO_MODE: limiting to %d frame(s)", len(valid_timepoints))
            valid_timepoint_set = set(valid_timepoints)
            logger.info(
                "Position %d, valid_timepoints = %d/%d",
                position,
                len(valid_timepoints),
                dataset_config.duration,
            )

            df_position = df_dataset_[
                (df_dataset_[Column.POSITION] == position)
                & (df_dataset_[Column.TIMEPOINT].isin(valid_timepoints))
            ].copy()
            if df_position.empty:
                continue
            df_position[Column.DATASET] = dataset_name

            # Crop grid
            crop_grid = build_crop_grid(df_position)
            num_crops = len(crop_grid)

            # For tracked crops, build a per-timepoint crop lookup dict
            # mapping timepoint -> (start_y, end_y, start_x, end_x, crop_ids)
            # so that crop coordinates can change frame-to-frame.  For grid
            # crops, the arrays are constant across all timepoints.
            if is_tracked:
                crop_size = int(
                    df_position[Column.DiffAEData.CROP_SIZE_X].iloc[0]
                    if Column.DiffAEData.CROP_SIZE_X in df_position.columns
                    else 128
                )
                tracked_crops: dict[int, tuple] = {}
                for t, grp in df_position.groupby(Column.TIMEPOINT):
                    sx_ = grp[Column.DiffAEData.START_X].values.astype(int)
                    sy_ = grp[Column.DiffAEData.START_Y].values.astype(int)
                    ex_ = sx_ + crop_size
                    ey_ = sy_ + crop_size
                    ci_ = grp[Column.CROP_INDEX].values
                    tracked_crops[int(t)] = (sy_, ey_, sx_, ex_, ci_)
            else:
                start_x = crop_grid[Column.DiffAEData.START_X].values.astype(int)
                start_y = crop_grid[Column.DiffAEData.START_Y].values.astype(int)
                end_x = crop_grid[Column.DiffAEData.END_X].values.astype(int)
                end_y = crop_grid[Column.DiffAEData.END_Y].values.astype(int)
                crop_ids = crop_grid[Column.CROP_INDEX].values

            # Z-project
            zarr_path = get_zarr_location_for_position(
                dataset_config,
                position,
            )
            image_dask = load_image(zarr_path, channels=[channel], level=level, compute=False)
            z_axis = DIMENSION_ORDER.index("Z")
            z_projection = (
                da.log(image_dask.std(axis=z_axis) + 1e-12)
                if is_bf
                else image_dask.max(axis=z_axis)
            )

            # Frame pairs
            frame_pairs = [
                (t, t + d, d)
                for d in range(1, max_dt + 1)
                for t in valid_timepoints
                if (t + d) in valid_timepoint_set
            ]
            needed_timepoints = sorted({t for pair in frame_pairs for t in pair[:2]})

            # Cache frames
            cache_start = time.time()
            needed_indices = sorted(needed_timepoints)
            needed_frames = z_projection[needed_indices, 0].compute(
                scheduler="threads", num_workers=n_io_workers
            )
            frame_cache: dict[int, np.ndarray] = {}
            for j, t in enumerate(needed_indices):
                frame = needed_frames[j].astype(np.float32, copy=False)
                if is_bf:
                    clip_low, clip_high = np.percentile(frame, [0.1, 99.9])
                    frame = np.clip(frame, clip_low, clip_high)
                    std = frame.std()
                    frame = (frame - frame.mean()) / (std if std > 0 else 1.0)
                else:
                    clip_low, clip_high = np.percentile(frame, [10, 98])
                    frame = np.clip(frame, clip_low, clip_high)
                    frame = (frame - clip_low) / (clip_high - clip_low + 1e-8) * 2.0 - 1.0
                frame_cache[t] = frame
            del needed_frames
            logger.info("Cached %d frames in %.1fs", len(frame_cache), time.time() - cache_start)

            # Intensity threshold
            intensity_threshold = (
                float(
                    np.percentile(
                        np.concatenate([f.ravel()[::10] for f in frame_cache.values()]),
                        intensity_pctl,
                    )
                )
                if intensity_pctl > 0
                else -float("inf")
            )
            logger.info(
                "threshold(%d-pctl)=%.6f pairs=%d crops=%d",
                intensity_pctl,
                intensity_threshold,
                len(frame_pairs),
                num_crops,
            )

            # Bind constant flow parameters once via partial.
            _compute_flow = partial(
                compute_image_pair_flow,
                attachment=attachment,
                compute_block_coherence=compute_block_coherence,
                compute_fast_coherence=compute_fast_coherence,
                compute_radial_coherence=compute_radial_coherence,
                speed_threshold=speed_threshold,
            )

            # Compute flow
            records: list[dict] = []
            if flow_scope == "image":
                desc = f"{dataset_name} pos={position}"
                if is_tracked:
                    desc += " (tracked)"
                with ThreadPoolExecutor(max_workers=n_io_workers) as pool:
                    futures: dict = {}
                    for t0, t1, dt in frame_pairs:
                        if is_tracked:
                            crops = tracked_crops.get(t0)
                            if crops is None:
                                continue
                            sy_, ey_, sx_, ex_, ci_ = crops
                        else:
                            sy_, ey_, sx_, ex_, ci_ = (
                                start_y,
                                end_y,
                                start_x,
                                end_x,
                                crop_ids,
                            )
                        futures[
                            pool.submit(
                                _compute_flow,
                                frame_cache[t0],
                                frame_cache[t1],
                                sy_,
                                ey_,
                                sx_,
                                ex_,
                                ci_,
                                t0,
                                dt,
                                intensity_threshold,
                            )
                        ] = (t0, dt)
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=desc,
                    ):
                        records.extend(future.result())
            else:
                # Crop scope: TVL1 per crop (grid only; tracked + crop
                # is rejected before the loop).
                crop_flow_args = [
                    (
                        frame_cache[t0][start_y[i] : end_y[i], start_x[i] : end_x[i]].copy(),
                        frame_cache[t1][start_y[i] : end_y[i], start_x[i] : end_x[i]].copy(),
                        crop_ids[i],
                        t0,
                        dt,
                        intensity_threshold,
                        attachment,
                        compute_block_coherence,
                        compute_fast_coherence,
                        compute_radial_coherence,
                        speed_threshold,
                    )
                    for t0, t1, dt in frame_pairs
                    for i in range(num_crops)
                ]
                num_jobs = n_jobs or os.cpu_count() // 6
                records = Parallel(n_jobs=num_jobs, backend="loky")(
                    delayed(compute_crop_flow)(*a)
                    for a in tqdm(crop_flow_args, desc=f"{dataset_name} pos={position}")
                )

            if visualize_optical_flow:
                plot_demo_summary(
                    frame_cache,
                    crop_grid,
                    dataset_name,
                    position,
                    intensity_threshold,
                    results_dir,
                    [channel],
                    flow_scope,
                    attachment,
                    compute_block_coherence,
                )

            frame_cache.clear()

            # Pivot & merge
            df_flow_pivoted = pivot_flow_records(records)
            df_position = df_position.merge(
                df_flow_pivoted,
                left_on=[Column.CROP_INDEX, Column.TIMEPOINT],
                right_on=["crop_index", "timepoint"],
                how="left",
            ).drop(columns=["crop_index_y", "timepoint_y"], errors="ignore")
            # rename back if pandas suffixed them
            if "crop_index_x" in df_position.columns:
                df_position.rename(columns={"crop_index_x": Column.CROP_INDEX}, inplace=True)
            if "timepoint_x" in df_position.columns:
                df_position.rename(columns={"timepoint_x": Column.TIMEPOINT}, inplace=True)

            for col in flow_columns:
                if col not in df_position.columns:
                    df_position[col] = np.nan

            # --- EMA smoothing of coherence metrics per crop ---
            df_position = df_position.sort_values([Column.CROP_INDEX, Column.TIMEPOINT])

            ema_stems_to_smooth = build_ema_stems(compute_fast_coherence, compute_radial_coherence)

            for alpha in ema_alphas:
                alpha_tag = str(alpha).replace(".", "")
                for d in range(1, max_dt + 1):
                    for stem in ema_stems_to_smooth:
                        raw_col = f"{stem}_dt{d}"
                        ema_col = f"ema{alpha_tag}_{stem}_dt{d}"
                        if raw_col in df_position.columns:
                            df_position[ema_col] = df_position.groupby(Column.CROP_INDEX)[
                                raw_col
                            ].transform(lambda s, a=alpha: s.ewm(alpha=a, adjust=False).mean())

            # --- Crop coherence time series diagnostic ---
            if visualize_optical_flow:
                plot_tracked_crop_coherence_timeseries(
                    df_position,
                    ds_name=dataset_name,
                    position=position,
                    out_dir=results_dir,
                    ema_alphas=ema_alphas,
                    compute_fast_coherence=compute_fast_coherence,
                    compute_radial_coherence=compute_radial_coherence,
                    max_crops=DEMO_MAX_TRACKED_CROPS_TO_PLOT,
                    max_dt=max_dt,
                )

            logger.info(
                "Position %d done in %.1fs for %d records",
                position,
                time.time() - position_start,
                len(records),
            )
            dataset_parts.append(df_position)

        # if list is not empty, concat, save, and optionally upload before
        # moving to the next dataset
        if dataset_parts:
            df_dataset_out = pd.concat(dataset_parts, ignore_index=True)
            parquet_path = make_name_unique(
                output_dir / f"{dataset_name}_optical_flow_dataframe.parquet"
            )
            df_dataset_out.to_parquet(parquet_path, index=False)
            logger.info("Saved parquet locally to [ %s ]", parquet_path)
            # If upload_to_fms is True, upload the parquet file to FMS and
            # register the FMS ID in the manifest; otherwise, register the local
            # path in the manifest
            if upload_to_fms:
                fms_annotations = build_fms_annotations(
                    dataset_config,
                    additional_notes=f"Optical flow features computed with {__file__}",
                )
                fms_id = upload_file_to_fms(parquet_path, fms_annotations, "parquet")
                logger.info("Uploaded optical flow features to FMS with ID [ %s ]", fms_id)
                optical_flow_manifest.locations[dataset_name] = DataframeLocation(fmsid=fms_id)
            elif (
                optical_flow_manifest.locations.get(dataset_name) is None
                or optical_flow_manifest.locations[dataset_name].fmsid is None
            ):
                # if dataset is not in manifest or has no FMS ID, register local
                # path (even if upload_to_fms is False) so that results are
                # accessible for downstream workflows
                optical_flow_manifest.locations[dataset_name] = DataframeLocation(path=parquet_path)
            save_dataframe_manifest(optical_flow_manifest)
        logger.info("Dataset done in [ %.1fs ]", time.time() - dataset_start)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
