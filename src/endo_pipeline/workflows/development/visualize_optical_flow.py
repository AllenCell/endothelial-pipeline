from endo_pipeline.cli import Datasets, PatchType
from endo_pipeline.settings.optical_flow import DEFAULT_EMA_ALPHA, DEFAULT_OPTICAL_FLOW_MAX_DT


def main(
    datasets: Datasets | None = None,
    patch_type: PatchType = "grid_based",
    max_dt: int = DEFAULT_OPTICAL_FLOW_MAX_DT,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> None:
    """
    Visualize TVL1 optical flow features for crops.

    #optical-flow #cell-centered #grid-based #visualization #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-optical-flow -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-optical-flow --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will visualize
    optical flow features for the first position of the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to visualize.
    patch_type
        Patch type to use for computing features.
    channel
        Imaging channel to use for computing features.
    max_dt
        Maximum temporal gap (inclusive).
    ema_alpha
        EMA smoothing alpha value for temporal coherence smoothing.
    """

    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.optical_flow import (
        OpticalFlowImagePair,
        calculate_optical_flow_intensity_threshold,
    )
    from endo_pipeline.library.process.image_processing import load_processed_bf_std_dev_image
    from endo_pipeline.library.visualize.optical_flow import (
        plot_optical_flow_coherence_over_time,
        plot_optical_flow_summary,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL
    from endo_pipeline.settings.optical_flow import (
        DEFAULT_OPTICAL_FLOW_COLLECTION,
        DEMO_MAX_TRACKED_CROPS_TO_PLOT,
        OPTICAL_FLOW_ATTACHMENT,
        OPTICAL_FLOW_MANIFEST_NAME_PREFIX,
        OPTICAL_FLOW_PERCENTILE,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    datasets = datasets or get_datasets_in_collection(DEFAULT_OPTICAL_FLOW_COLLECTION)

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset and one position")
        datasets = datasets[:1]
        max_positions = 1
    else:
        max_positions = None

    # Set channel-aware options
    intensity_percentile = OPTICAL_FLOW_PERCENTILE
    attachment = OPTICAL_FLOW_ATTACHMENT

    # Load optical flow dataframe manifest
    name_prefix = OPTICAL_FLOW_MANIFEST_NAME_PREFIX
    name_suffix = f"_{patch_type}"
    manifest = load_dataframe_manifest(f"{name_prefix}{name_suffix}")

    # Select sorting column
    feature_column = Column.OpticalFlow.UNIT_VECTOR_MEAN
    pick_labels = ["COHERENT", "MEDIAN", "INCOHERENT"]

    for dataset_name in datasets:
        logger.info("Starting optical flow visualization for dataset '%s'", dataset_name)

        dataset_config = load_dataset_config(dataset_name)

        # Load optical flow features for dataset
        location = get_dataframe_location_for_dataset(manifest, dataset_name)
        df = load_dataframe(location)

        # Find unique positions for datasets
        unique_positions = df[Column.POSITION].unique()
        if max_positions is not None:
            unique_positions = unique_positions[:max_positions]

        for position in unique_positions:
            output_name = f"{dataset_name}_P{position}{name_suffix}"

            # Select three representative examples
            df_position = df[df[Column.POSITION] == position].sort_values(feature_column).dropna()
            crop_picks = [
                df_position.iloc[-1],
                df_position.iloc[len(df_position) // 2],
                df_position.iloc[0],
            ]

            # Build frame pairs from timepoint sets
            requested_timepoints = [pick[Column.TIMEPOINT] for pick in crop_picks]
            image_pairs = [
                OpticalFlowImagePair(t0=t0, t1=t0 + dt, dt=dt)
                for dt in range(1, max_dt + 1)
                for t0 in requested_timepoints
            ]
            needed_timepoints = sorted({t for pair in image_pairs for t in pair[:2]})

            # Cache frames for required timepoints
            image_cache: dict[int, np.ndarray] = {}
            for timepoint in needed_timepoints:
                image_cache[timepoint] = (
                    load_processed_bf_std_dev_image(
                        dataset_config, position, [timepoint], DIFFAE_ZARR_RESOLUTION_LEVEL
                    )
                    .squeeze()
                    .compute()
                )

            # Calculate intensity threshold based on intensity percentile
            intensity_threshold = calculate_optical_flow_intensity_threshold(
                intensity_percentile, list(image_cache.values())
            )

            # Plot optical flow summary for picks
            plot_optical_flow_summary(
                crop_picks=crop_picks,
                image_pairs=image_pairs,
                pick_labels=pick_labels,
                image_cache=image_cache,
                feature_data=df_position,
                output_name=output_name,
                output_dir=output_path,
                attachment=attachment,
                intensity_threshold=intensity_threshold,
            )

            # Plot optical flow coherence over time
            plot_optical_flow_coherence_over_time(
                df_position,
                output_name=output_name,
                output_dir=output_path,
                ema_alpha=ema_alpha,
                max_crops=DEMO_MAX_TRACKED_CROPS_TO_PLOT,
                max_dt=max_dt,
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
