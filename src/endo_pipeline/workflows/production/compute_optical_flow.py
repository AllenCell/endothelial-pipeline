from typing import Literal

from endo_pipeline.cli import CropPattern, Datasets, FloatList
from endo_pipeline.settings.optical_flow import (
    DEFAULT_EMA_ALPHAS,
    DEFAULT_OPTICAL_FLOW_MAX_DT,
    DEFAULT_SPEED_THRESHOLD,
)


def main(
    datasets: Datasets | None = None,
    crop_pattern: CropPattern = "grid",
    channel: Literal["BF", "EGFP"] = "BF",
    max_dt: int = DEFAULT_OPTICAL_FLOW_MAX_DT,
    ema_alphas: FloatList = list(DEFAULT_EMA_ALPHAS),
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    num_workers: int = 1,
) -> None:
    """
    Compute TVL1 optical flow features for crops.

    This workflow compute TVL1 optical flow between frame pairs at temporal gaps
    dt = 1, 2, ..., max_dt for every crop and timepoint. Pixels whose intensity
    falls below a channel-aware percentile threshold are masked before computing
    per-crop summary statistics.

    The following settings are channel-aware:

    | Setting             | Channel = BF   | Channel = EGFP   |
    | ------------------- | -------------- | ---------------- |
    | Intensity threshold | 0              | 95               |
    | TVL1 attachment     | 2.5            | 7.5              |
    | Z-projection        | std(axis=Z)    | max(axis=Z)      |
    | Transformation      | log(x + 1e-12) | None             |
    | Clipping            | [0.1, 99.9]    | [10, 98]         |
    | Normalization       | z-score        | scale to [-1, 1] |

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-optical-flow -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-optical-flow --datasets DATASET_NAME
    ```

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
    crop_pattern
        Crop pattern to use for computing features.
    channel
        Imaging channel to use for computing features.
    max_dt
        Maximum temporal gap (inclusive).
    ema_alphas
        EMA smoothing alpha values for temporal coherence smoothing.
    speed_threshold
        Minimum pixel speed for the "fast" coherence features.
    num_workers
        Number of worker processes to use.
    """

    import logging
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from functools import partial

    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_dataframe,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.optical_flow import (
        OpticalFlowImagePair,
        build_image_pair_crops_for_grid,
        build_image_pair_crops_for_tracked,
        build_merged_optical_flow_dataframe,
        calculate_optical_flow_intensity_threshold,
        compute_image_pair_flow,
    )
    from endo_pipeline.library.visualize.supplemental_movies import (
        load_bf_std_dev_image,
        load_egfp_image,
    )
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
        OPTICAL_FLOW_CHANNEL_ATTACHMENT,
        OPTICAL_FLOW_CHANNEL_PERCENTILE,
        OPTICAL_FLOW_COLUMNS_TO_COMPUTE,
        OPTICAL_FLOW_MANIFEST_NAME_PREFIX,
    )
    from endo_pipeline.settings.workflow_defaults import FEATURES_UNFILTERED_MANIFEST_NAMES

    # Pin OpenMP to 1 thread per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    datasets = datasets or get_datasets_in_collection(DEFAULT_OPTICAL_FLOW_COLLECTION)

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset, one position, and 10 timepoints")
        datasets = datasets[:1]
        max_positions = 1
        max_timepoints = 20
    else:
        max_positions = None
        max_timepoints = None

    # Set channel-aware options
    intensity_percentile = OPTICAL_FLOW_CHANNEL_PERCENTILE[channel]
    attachment = OPTICAL_FLOW_CHANNEL_ATTACHMENT[channel]
    image_loader = load_bf_std_dev_image if channel == "BF" else load_egfp_image

    # Load dataframe with DiffAE feature metadata (no filtering yet) to get crop
    # coordinates and timepoints for each dataset/position.
    feature_dataframe_manifest_name = FEATURES_UNFILTERED_MANIFEST_NAMES[crop_pattern]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)
    columns_to_compute = list(OPTICAL_FLOW_COLUMNS_TO_COMPUTE[crop_pattern])

    # Load or create optical flow manifest and set parameters before the loop so
    # that it is available even if the workflow fails partway through. Add
    # suffix to manifest name in demo mode to avoid overwriting real results
    # with partial demo results.
    name_prefix = OPTICAL_FLOW_MANIFEST_NAME_PREFIX
    name_suffix = f"_{channel.lower()}_{crop_pattern}{'_demo' if DEMO_MODE else ''}"
    optical_flow_manifest = create_dataframe_manifest(
        f"{name_prefix}{name_suffix}",
        workflow_name=__file__,
    )
    optical_flow_manifest.parameters = {
        "crop_pattern": crop_pattern,
        "channel": channel,
        "max_dt": max_dt,
        "intensity_percentile": intensity_percentile,
        "attachment": attachment,
        "ema_alphas": list(ema_alphas),
        "speed_threshold": speed_threshold,
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
            df_position = df_dataset_[
                (df_dataset_[Column.POSITION] == position)
                & (df_dataset_[Column.TIMEPOINT].isin(valid_timepoints))
            ].copy()

            # If there are no crops for the given position and timepoints, skip
            if df_position.empty:
                continue

            # Add dataset name to dataframe
            df_position[Column.DATASET] = dataset_name

            # For tracked crops, build a per-timepoint crop lookup dict
            # mapping timepoint -> (start_y, end_y, start_x, end_x, crop_ids)
            # so that crop coordinates can change frame-to-frame. For grid
            # crops, the arrays are constant across all timepoints.
            if crop_pattern == "tracked":
                get_crops_for_timepoint = build_image_pair_crops_for_tracked(df_position)
            else:
                get_crops_for_timepoint = build_image_pair_crops_for_grid(df_position)

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
                    image_loader(
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
                thresh=intensity_threshold,
                attachment=attachment,
                speed_threshold=speed_threshold,
            )

            records: list[dict] = []

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
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
                df_position, records, max_dt, ema_alphas
            )

            position_dataframes.append(df_position)

        if not position_dataframes:
            continue

        # Save dataframe to file
        df_dataset_out = pd.concat(position_dataframes, ignore_index=True)
        save_path = output_path / f"{name_prefix}_{dataset_name}{name_suffix}.parquet"
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
