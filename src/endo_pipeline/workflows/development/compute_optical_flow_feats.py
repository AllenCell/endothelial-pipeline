import logging
from typing import Literal

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL
from endo_pipeline.settings.optical_flow import DEFAULT_OPTICAL_FLOW_MAX_DT, NUM_IO_WORKERS

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
    upload_to_fms: bool = False,
    visualize_optical_flow: bool = False,
    include_cell_piling: bool = False,
    include_pre_steady_state: bool = False,
    include_all_conditions: bool = False,
    compute_block_coherence: bool = False,
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
        Position indices to process (e.g. ``[0, 1]``).
        ``None`` processes all positions in each dataset.
    channel
        Imaging channel to load (``"BF"`` or ``"EGFP"``).
    level
        Zarr resolution level.  Defaults to ``DIFFAE_ZARR_RESOLUTION_LEVEL``.
    annotations_to_exclude
        Explicit list of timepoint annotations to exclude.  When ``None``
        (default), the list is built from *include_cell_piling* and
        *include_pre_steady_state*.  Pass an empty list ``[]`` to
        bypass **all** annotation filtering (including quality
        annotations).  Passing a non-empty list overrides both boolean
        flags.
    max_dt
        Maximum temporal gap (inclusive).  Defaults to ``DEFAULT_OPTICAL_FLOW_MAX_DT``.
    intensity_percentile
        Pixels below this percentile (computed across all cached frames)
        are masked out.  None -> auto-select based on channel
        (EGFP -> 95, BF -> 0).
    n_jobs
        Parallel workers used in "crop" flow scope (joblib/loky).
    n_io_workers
        Concurrent I/O workers for dask frame loading and
        ``ThreadPoolExecutor`` in image-scope flow.  Default ``_IO_WORKERS`` (16).
    flow_scope
        "image" or "crop" (see Flow scopes above).
    upload_to_fms
        If True, save parquet, upload to FMS, and register in the
        dataframe manifest.  Default False to prevent accidental
        uploads during development or external use.
    visualize_optical_flow
        If True, produce diagnostic plots (R/G composite, quiver,
        speed & angle histograms) for one randomly chosen crop per
        (dataset, position) pair.  Saved to results/optical_flow/.
    include_cell_piling
        If True, retain timepoints annotated as cell-piling.
        Default False (they are excluded).
    include_pre_steady_state
        If True, retain timepoints before visual steady state.
        Default False (they are excluded).
    include_all_conditions
        If True, bypass **all** annotation filtering — including quality
        annotations (scope errors, temp artifacts, XY/Z shifts, unfed).
        Overrides both *include_cell_piling* and *include_pre_steady_state*.
    compute_block_coherence
        If True, compute multi-scale block-averaged coherence statistics
        (``optical_flow_angle_std_box{N}``) for each box size.  Off by
        default to save time.
    """
    import os
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
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_image,
        make_name_unique,
        upload_file_to_fms,
    )
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
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        get_feature_dataframe_manifest_name,
        get_zarr_location_for_position,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings import DIMENSION_ORDER
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_FEATURE_COLUMN_NAMES
    from endo_pipeline.settings.optical_flow import (
        DEFAULT_OMP_NUM_THREADS,
        DEFAULT_OPENBLAS_NUM_THREADS,
        DEFAULT_OPTICAL_FLOW_COLLECTION,
        DEFAULT_OPTICAL_FLOW_MANIFEST_NAME,
        DEMO_MAX_DATASETS,
        DEMO_MAX_FRAMES,
        DEMO_MAX_POSITIONS,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    # Pin OpenMP to 1 thread per worker
    os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_OMP_NUM_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", DEFAULT_OPENBLAS_NUM_THREADS)

    diffae_columns_to_drop = list(DIFFAE_FEATURE_COLUMN_NAMES)

    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_OPTICAL_FLOW_COLLECTION)

    if DEMO_MODE:
        datasets = datasets[:DEMO_MAX_DATASETS]
        upload_to_fms = False
        visualize_optical_flow = True
        logger.info(
            "DEMO_MODE is ON. Processing %d dataset(s), %d position(s) each, upload disabled.",
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
    flow_columns = build_optical_flow_feature_cols(max_dt, compute_block_coherence)
    is_bf = channel == "BF"

    logger.info(
        "Optical-flow extraction --> scope = %s | dt = 1 .. %d | percentile = %d | attachment = %.1f | channel = %s | block_coherence = %s",
        flow_scope,
        max_dt,
        intensity_pctl,
        attachment,
        channel,
        compute_block_coherence,
    )

    # Shared manifests
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    dataframe_name = get_feature_dataframe_manifest_name(
        model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_name)

    # Load or create optical flow manifest and set parameters before the loop so
    # that it is available even if the workflow fails partway through. Add
    # suffix to manifest name in demo mode to avoid overwriting real results
    # with partial demo results.
    demo_suffix = "_demo" if DEMO_MODE else ""
    optical_flow_manifest = create_dataframe_manifest(
        f"{DEFAULT_OPTICAL_FLOW_MANIFEST_NAME}{demo_suffix}",
        workflow_name=__file__,
    )
    optical_flow_manifest.parameters = {
        "channel": channel,
        "level": level,
        "max_dt": max_dt,
        "flow_scope": flow_scope,
        "intensity_percentile": intensity_pctl,
        "attachment": attachment,
        "compute_block_coherence": compute_block_coherence,
        "annotations_excluded": [a.value for a in annotations_to_exclude],
    }
    save_dataframe_manifest(optical_flow_manifest)

    # set output directory for dataframes
    output_dir = get_output_path("optical_flow", "dataframes")

    for dataset_idx, dataset_name in enumerate(datasets, 1):
        dataset_start = time.time()
        logger.info("Dataset %d/%d: %s", dataset_idx, len(datasets), dataset_name)

        dataset_config = load_dataset_config(dataset_name)
        df_dataset = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=None,
            filter_by_annotations=False,
        )

        position_list = sorted(df_dataset[ColumnName.POSITION].unique().tolist())
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

            df_position = df_dataset[
                (df_dataset[ColumnName.POSITION] == position)
                & (df_dataset[ColumnName.TIMEPOINT].isin(valid_timepoints))
            ].copy()
            if df_position.empty:
                continue
            df_position["dataset"] = dataset_name

            # Crop grid
            crop_grid = build_crop_grid(df_position)
            start_x = crop_grid[ColumnName.DiffAEData.START_X].values.astype(int)
            start_y = crop_grid[ColumnName.DiffAEData.START_Y].values.astype(int)
            end_x = crop_grid[ColumnName.DiffAEData.END_X].values.astype(int)
            end_y = crop_grid[ColumnName.DiffAEData.END_Y].values.astype(int)
            crop_ids = crop_grid[ColumnName.CROP_INDEX].values
            num_crops = len(crop_grid)

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

            # Compute flow
            records: list[dict] = []
            if flow_scope == "image":

                def _image_pair(
                    t0,
                    t1,
                    dt,
                    _cache=frame_cache,
                    _start_y=start_y,
                    _end_y=end_y,
                    _start_x=start_x,
                    _end_x=end_x,
                    _crop_ids=crop_ids,
                    _threshold=intensity_threshold,
                    _attachment=attachment,
                    _compute_block=compute_block_coherence,
                ):
                    frame_0, frame_1 = _cache[t0], _cache[t1]
                    return compute_image_pair_flow(
                        frame_0,
                        frame_1,
                        _start_y,
                        _end_y,
                        _start_x,
                        _end_x,
                        _crop_ids,
                        t0,
                        dt,
                        _threshold,
                        _attachment,
                        _compute_block,
                    )

                with ThreadPoolExecutor(max_workers=n_io_workers) as pool:
                    futures = {
                        pool.submit(_image_pair, t0, t1, dt): (t0, dt) for t0, t1, dt in frame_pairs
                    }
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"{dataset_name} pos={position}",
                    ):
                        records.extend(future.result())
            else:
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
                left_on=[ColumnName.CROP_INDEX, ColumnName.TIMEPOINT],
                right_on=["crop_index", "timepoint"],
                how="left",
            ).drop(columns=["crop_index", "timepoint"], errors="ignore")

            for col in flow_columns:
                if col not in df_position.columns:
                    df_position[col] = np.nan
            df_position.drop(
                columns=[c for c in diffae_columns_to_drop if c in df_position.columns],
                inplace=True,
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
                if optical_flow_manifest.locations[dataset_name].path != str(parquet_path):
                    # if local path has changed (e.g. due to new output_dir), update
                    # manifest; this is important to ensure that the manifest points
                    # to the correct location even if upload_to_fms is False
                    logger.warning(
                        "Local path for dataset %s has changed from [ %s ] to [ %s ]",
                        dataset_name,
                        optical_flow_manifest.locations[dataset_name].path,
                        parquet_path,
                    )
                optical_flow_manifest.locations[dataset_name] = DataframeLocation(path=parquet_path)
            save_dataframe_manifest(optical_flow_manifest)
        logger.info("Dataset done in [ %.1fs ]", time.time() - dataset_start)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
