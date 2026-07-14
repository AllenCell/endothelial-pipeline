from endo_pipeline.cli import Datasets, PatchType
from endo_pipeline.settings.optical_flow import DEFAULT_EMA_ALPHA, DEFAULT_OPTICAL_FLOW_MAX_DT


def main(
    datasets: Datasets | None = None,
    patch_type: PatchType = "grid_based",
    max_dt: int = DEFAULT_OPTICAL_FLOW_MAX_DT,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> None:
    """
    Compute TVL1 optical flow features for crops.

    #optical-flow #cell-centered #grid-based #test-ready #workers

    This workflow compute TVL1 optical flow between frame pairs at temporal gaps
    dt = 1, 2, ..., max_dt for every crop and timepoint. Pixels whose intensity
    falls below a percentile threshold are masked before computing
    per-crop summary statistics (see `OPTICAL_FLOW_PERCENTILE` setting).

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-optical-flow -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-optical-flow --datasets DATASET_NAME
    ```

    ## Worker processes

    TVL1 is pinned to one thread per call (OMP_NUM_THREADS=1 and
    OPENBLAS_NUM_THREADS=1), so the bottleneck is NFS read throughput rather
    than CPU. We recommend `--num-workers=16` based on empirical testing when
    using compute hardware similar to the following: 512 GB RAM, 128 physical /
    256 logical CPU cores, 4x A100 80 GB GPUs. At 16 concurrent workers, each
    holding ~0.5 GB per frame, peak memory is ~8 GB while NFS throughput is
    fully saturated. Beyond 16 workers, the wall-clock time plateaus but memory
    grows linearly with no additional speedup.

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
    patch_type
        Patch type to use for computing features.
    max_dt
        Maximum temporal gap (inclusive).
    ema_alpha
        EMA smoothing alpha value for temporal coherence smoothing.
    """

    import logging
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from functools import partial

    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    from endo_pipeline.cli import DEMO_MODE, NUM_WORKERS, UPLOAD_TO_FMS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_dataframe,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.optical_flow import (
        OpticalFlowImagePair,
        build_image_pair_crops_for_cell_centered,
        build_image_pair_crops_for_grid_based,
        build_merged_optical_flow_dataframe,
        calculate_optical_flow_intensity_threshold,
        compute_image_pair_flow,
    )
    from endo_pipeline.library.process.image_processing import load_processed_bf_std_dev_image
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL
    from endo_pipeline.settings.optical_flow import (
        DEFAULT_OPTICAL_FLOW_COLLECTION,
        OPTICAL_FLOW_ATTACHMENT,
        OPTICAL_FLOW_COLUMNS_TO_COMPUTE,
        OPTICAL_FLOW_MANIFEST_NAMES,
        OPTICAL_FLOW_PERCENTILE,
    )
    from endo_pipeline.settings.workflow_defaults import FEATURES_UNFILTERED_MANIFEST_NAMES

    # Pin OpenMP to 1 thread per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    datasets = datasets or get_datasets_in_collection(DEFAULT_OPTICAL_FLOW_COLLECTION)

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset, one position, and 10 timepoints")
        datasets = datasets[:1]
        max_positions = 1
        max_timepoints = 10
    else:
        max_positions = None
        max_timepoints = None

    # Set channel-aware options
    intensity_percentile = OPTICAL_FLOW_PERCENTILE
    attachment = OPTICAL_FLOW_ATTACHMENT

    # Load dataframe with DiffAE feature metadata (no filtering yet) to get crop
    # coordinates and timepoints for each dataset/position.
    feature_dataframe_manifest_name = FEATURES_UNFILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)
    columns_to_compute = list(OPTICAL_FLOW_COLUMNS_TO_COMPUTE[patch_type])

    # Load or create optical flow manifest and set parameters before the loop so
    # that it is available even if the workflow fails partway through. Add
    # suffix to manifest name in demo mode to avoid overwriting real results
    # with partial demo results.
    manifest_name = OPTICAL_FLOW_MANIFEST_NAMES[patch_type]
    name_suffix = "_demo" if DEMO_MODE else ""
    optical_flow_manifest = create_dataframe_manifest(
        f"{manifest_name}{name_suffix}",
        workflow_name=__file__,
    )
    optical_flow_manifest.parameters = {
        "patch_type": patch_type,
        "max_dt": max_dt,
        "ema_alpha": ema_alpha,
    }
    save_dataframe_manifest(optical_flow_manifest)

    for dataset_name in datasets:
        logger.info("Starting optical flow computation for dataset '%s'", dataset_name)

        dataset_config = load_dataset_config(dataset_name)
        df_dataset = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df_dataset_: pd.DataFrame = df_dataset[columns_to_compute].compute()

        # Get list of valid positions and subset if necessary
        position_list = dataset_config.zarr_positions
        if max_positions is not None:
            position_list = position_list[:max_positions]

        position_dataframes: list[pd.DataFrame] = []

        for position in position_list:
            valid_timepoints = list(range(dataset_config.duration))
            if max_timepoints is not None:
                valid_timepoints = valid_timepoints[:max_timepoints]

            valid_timepoint_set = set(valid_timepoints)

            # Filter dataframe down to selected position and timepoint
            df_position = (
                df_dataset_[
                    (df_dataset_[Column.POSITION] == position)
                    & (df_dataset_[Column.TIMEPOINT].isin(valid_timepoints))
                ]
                .copy()
                .dropna()
            )

            # If there are no crops for the given position and timepoints, skip
            if df_position.empty:
                continue

            # For tracked crops, build a per-timepoint crop lookup dict
            # mapping timepoint -> (start_y, end_y, start_x, end_x, crop_ids)
            # so that crop coordinates can change frame-to-frame. For grid
            # crops, the arrays are constant across all timepoints.
            if patch_type == "cell_centered":
                get_crops_for_timepoint = build_image_pair_crops_for_cell_centered(df_position)
            else:
                get_crops_for_timepoint = build_image_pair_crops_for_grid_based(df_position)

            # Build frame pairs from timepoint sets
            image_pairs = [
                OpticalFlowImagePair(t0=t0, t1=t0 + dt, dt=dt)
                for dt in range(1, max_dt + 1)
                for t0 in valid_timepoints
                if (t0 + dt) in valid_timepoint_set
            ]
            needed_timepoints = sorted({t for pair in image_pairs for t in pair[:2]})

            # Cache frames for required timepoints
            cache_start = time.time()
            frame_cache: dict[int, np.ndarray] = {}
            for timepoint in needed_timepoints:
                frame_cache[timepoint] = (
                    load_processed_bf_std_dev_image(
                        dataset_config, position, [timepoint], DIFFAE_ZARR_RESOLUTION_LEVEL
                    )
                    .squeeze()
                    .compute()
                )
            logger.info("Cached %d frames in %.1fs", len(frame_cache), time.time() - cache_start)

            # Calculate intensity threshold based on intensity percentile
            intensity_threshold = calculate_optical_flow_intensity_threshold(
                intensity_percentile, list(frame_cache.values())
            )
            logger.info(
                "Intensity threshold (%d percentile) = %.6f",
                intensity_percentile,
                intensity_threshold,
            )

            # Bind constant flow parameters once via partial
            compute_image_pair_flow_partial = partial(
                compute_image_pair_flow,
                intensity_threshold=intensity_threshold,
                attachment=attachment,
            )

            records: list[dict] = []

            with ThreadPoolExecutor(max_workers=NUM_WORKERS or 1) as pool:
                futures = [
                    pool.submit(
                        compute_image_pair_flow_partial,
                        frame_cache[image_pair.t0],
                        frame_cache[image_pair.t1],
                        image_pair,
                        get_crops_for_timepoint(image_pair.t0),
                    )
                    for image_pair in image_pairs
                ]

                for future in tqdm(as_completed(futures), total=len(futures)):
                    records.extend(future.result())

            # Append optical flow features from records to the position
            # dataframe, add any missing columns, and apply EMA smoothing
            df_position = build_merged_optical_flow_dataframe(
                df_position, records, max_dt, ema_alpha
            )

            position_dataframes.append(df_position)

        if not position_dataframes:
            continue

        # Save dataframe to file
        df_dataset_out = pd.concat(position_dataframes, ignore_index=True)
        df_dataset_out[Column.DATASET] = dataset_name
        save_path = output_path / f"{manifest_name}_{dataset_name}{name_suffix}.parquet"
        df_dataset_out.to_parquet(save_path, index=False)

        # Create location object with output path
        location = optical_flow_manifest.locations.get(dataset_name, DataframeLocation())
        location.path = save_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(
                dataset_config,
                additional_notes="Optical flow features",
            )
            fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None

        # Add dataframe location to dataframe manifest and save
        optical_flow_manifest.locations[dataset_name] = location
        save_dataframe_manifest(optical_flow_manifest)

        logger.info("Finished computing optical flow for dataset '%s'", dataset_name)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
