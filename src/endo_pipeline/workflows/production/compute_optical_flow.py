import logging
from typing import Literal

from endo_pipeline.cli import CropPattern, Datasets, FloatList
from endo_pipeline.settings.optical_flow import (
    DEFAULT_EMA_ALPHAS,
    DEFAULT_OPTICAL_FLOW_MAX_DT,
    DEFAULT_SPEED_THRESHOLD,
    NUM_IO_WORKERS,
)

logger = logging.getLogger(__name__)


def main(  # noqa: C901
    datasets: Datasets | None = None,
    channel: Literal["BF", "EGFP"] = "BF",
    max_dt: int = DEFAULT_OPTICAL_FLOW_MAX_DT,
    n_io_workers: int = NUM_IO_WORKERS,
    crop_pattern: CropPattern = "grid",
    upload_to_fms: bool = False,
    visualize_optical_flow: bool = False,
    ema_alphas: FloatList = list(DEFAULT_EMA_ALPHAS),
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
) -> None:
    """Optical-flow feature extraction with multi-scale temporal coherence.

    Compute TVL1 optical flow between frame pairs at temporal gaps
    dt = 1, 2, ..., max_dt for every crop and timepoint.  Pixels whose
    intensity falls below a channel-aware percentile threshold are masked
    before computing per-crop summary statistics (mean/median speed,
    circular mean/std of angle, etc.).

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


    Output:
        Results are saved as one parquet file per dataset under
        results/optical_flow/ and (optionally) uploaded to FMS.

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will compute
    optical flow features for the first position and first 10 timepoints of the
    first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compute optical flow on.
    channel
        Imaging channel to load (``"BF"`` or ``"EGFP"``).
    max_dt
        Maximum temporal gap (inclusive).
    n_io_workers
        Concurrent I/O workers for dask frame loading and
        ``ThreadPoolExecutor`` in image-scope flow.
    crop_pattern
        "grid" or "tracked" (see Crop patterns above).
    upload_to_fms
        If True, save parquet, upload to FMS, and register in the
        dataframe manifest.
    visualize_optical_flow
        If True, produce diagnostic plots (R/G composite, quiver,
        speed & angle histograms) for one randomly chosen crop per
        (dataset, position) pair.  Saved to results/optical_flow/.
    ema_alphas
        EMA smoothing alpha values for temporal coherence smoothing.
    speed_threshold
        Minimum pixel speed for the "fast" coherence features.
    """

    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from functools import partial

    import dask.array as da
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
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
        build_optical_flow_feature_cols,
        compute_image_pair_flow,
        pivot_flow_records,
        plot_demo_summary,
        plot_tracked_crop_coherence_timeseries,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        get_zarr_location_for_position,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL, DIMENSION_ORDER
    from endo_pipeline.settings.optical_flow import (
        DEFAULT_OMP_NUM_THREADS,
        DEFAULT_OPENBLAS_NUM_THREADS,
        DEFAULT_OPTICAL_FLOW_COLLECTION,
        DEFAULT_OPTICAL_FLOW_MANIFEST_NAME,
        DEMO_MAX_TRACKED_CROPS_TO_PLOT,
        OPTICAL_FLOW_CHANNEL_ATTACHMENT,
        OPTICAL_FLOW_CHANNEL_PERCENTILE,
        OPTICAL_FLOW_COLUMNS_TO_COMPUTE,
        OPTICAL_FLOW_EMA_STEMS,
    )
    from endo_pipeline.settings.workflow_defaults import FEATURES_UNFILTERED_MANIFEST_NAMES

    # Pin OpenMP to 1 thread per worker
    os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_OMP_NUM_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", DEFAULT_OPENBLAS_NUM_THREADS)

    is_tracked = crop_pattern == "tracked"

    datasets = datasets or get_datasets_in_collection(DEFAULT_OPTICAL_FLOW_COLLECTION)

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset, one position, and 10 timepoints")
        datasets = datasets[:1]
        max_positions = 1
        max_timepoints = 20
    else:
        max_positions = None
        max_timepoints = None

    results_dir = get_output_path("optical_flow", "plots")

    # Set channel-aware options
    intensity_percentile = OPTICAL_FLOW_CHANNEL_PERCENTILE[channel]
    attachment = OPTICAL_FLOW_CHANNEL_ATTACHMENT[channel]

    flow_columns = build_optical_flow_feature_cols(
        max_dt,
        ema_alphas=ema_alphas,
    )
    is_bf = channel == "BF"

    logger.info(
        "Optical-flow extraction --> crop_pattern=%s | dt=1..%d "
        "| percentile=%d | attachment=%.1f | channel=%s "
        "| ema_alphas=%s | speed_threshold=%.2f",
        crop_pattern,
        max_dt,
        intensity_percentile,
        attachment,
        channel,
        ema_alphas,
        speed_threshold,
    )

    # Load dataframe with diffae feature metadata (no filtering yet) to get crop
    # coordinates and timepoints for each dataset/position.
    feature_dataframe_manifest_name = FEATURES_UNFILTERED_MANIFEST_NAMES[crop_pattern]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # For tracked crops, we also need TRACK_ID, CROP_SIZE_X
    columns_to_compute = list(OPTICAL_FLOW_COLUMNS_TO_COMPUTE[crop_pattern])

    # Load or create optical flow manifest and set parameters before the loop so
    # that it is available even if the workflow fails partway through. Add
    # suffix to manifest name in demo mode to avoid overwriting real results
    # with partial demo results.
    demo_suffix = "_demo" if DEMO_MODE else ""
    optical_flow_manifest = create_dataframe_manifest(
        f"{DEFAULT_OPTICAL_FLOW_MANIFEST_NAME}_{crop_pattern}{demo_suffix}",
        workflow_name=__file__,
    )
    optical_flow_manifest.parameters = {
        "channel": channel,
        "max_dt": max_dt,
        "crop_pattern": crop_pattern,
        "intensity_percentile": intensity_percentile,
        "attachment": attachment,
        "ema_alphas": list(ema_alphas),
        "speed_threshold": speed_threshold,
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

        # Get list of valid positions and subset if necessary
        position_list = dataset_config.zarr_positions
        if max_positions is not None:
            position_list = position_list[:max_positions]

        dataset_parts: list[pd.DataFrame] = []

        for position in position_list:
            position_start = time.time()

            valid_timepoints = list(range(dataset_config.duration))
            if max_timepoints is not None:
                valid_timepoints = valid_timepoints[:max_timepoints]

            valid_timepoint_set = set(valid_timepoints)

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
            image_dask = load_image(
                zarr_path, channels=[channel], level=DIFFAE_ZARR_RESOLUTION_LEVEL, compute=False
            )
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
                        intensity_percentile,
                    )
                )
                if intensity_percentile > 0
                else -float("inf")
            )
            logger.info(
                "threshold(%d-pctl)=%.6f pairs=%d crops=%d",
                intensity_percentile,
                intensity_threshold,
                len(frame_pairs),
                num_crops,
            )

            # Bind constant flow parameters once via partial.
            _compute_flow = partial(
                compute_image_pair_flow,
                attachment=attachment,
                speed_threshold=speed_threshold,
            )

            # Compute flow
            records: list[dict] = []
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

            if visualize_optical_flow:
                plot_demo_summary(
                    frame_cache,
                    crop_grid,
                    dataset_name,
                    position,
                    intensity_threshold,
                    results_dir,
                    [channel],
                    attachment,
                )

            frame_cache.clear()

            # Pivot & merge
            df_flow_pivoted = pivot_flow_records(records)
            df_position = df_position.merge(
                df_flow_pivoted,
                left_on=[Column.CROP_INDEX, Column.TIMEPOINT],
                right_on=[Column.CROP_INDEX, Column.TIMEPOINT],
                how="left",
            ).drop(columns=[f"{Column.CROP_INDEX}_y", f"{Column.TIMEPOINT}_y"], errors="ignore")
            # rename back if pandas suffixed them
            if f"{Column.CROP_INDEX}_x" in df_position.columns:
                df_position.rename(
                    columns={f"{Column.CROP_INDEX}_x": Column.CROP_INDEX}, inplace=True
                )
            if f"{Column.TIMEPOINT}_x" in df_position.columns:
                df_position.rename(
                    columns={f"{Column.TIMEPOINT}_x": Column.TIMEPOINT}, inplace=True
                )

            for col in flow_columns:
                if col not in df_position.columns:
                    df_position[col] = np.nan

            # --- EMA smoothing of coherence metrics per crop ---
            df_position = df_position.sort_values([Column.CROP_INDEX, Column.TIMEPOINT])

            for alpha in ema_alphas:
                alpha_tag = str(alpha).replace(".", "")
                for d in range(1, max_dt + 1):
                    for stem in OPTICAL_FLOW_EMA_STEMS:
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
