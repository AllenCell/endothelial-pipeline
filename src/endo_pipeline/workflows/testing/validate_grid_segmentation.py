from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None):
    """
    Validate grid-based crop positions match grid segmentations.

    #validation #grid-based #workers

    This workflow compares the crop locations in the grid segmentations produced
    by the `create_grid_segmentation` workflow with the crop locations listed in
    the grid-based feature dataframe produced by the `calculate_pca_features`
    workflow for selected datasets.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe validate-grid-segmentation -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe validate-grid-segmentation --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `live_cdh5_seg_based_feat_datasets` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run
    validation on the first two positions and first 10 timepoints of a single
    dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to validate.
    """

    import logging

    from skimage.measure import regionprops

    from endo_pipeline.cli import DEMO_MODE, NUM_WORKERS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import load_dataframe, load_image
    from endo_pipeline.library.process.general_image_preprocessing import (
        ImageProcessingArgs,
        build_analysis_queue,
        process_task_queue,
    )
    from endo_pipeline.library.process.progress_bar import ProgressBar
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_image_location_for_dataset,
        load_dataframe_manifest,
        load_image_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.workflow_defaults import (
        GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    dataset_names = datasets or get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    image_manifest = load_image_manifest("grid_seg_zarr")
    feature_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME)

    columns_to_compute = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.CROP_INDEX,
        Column.DiffAEData.START_Y,
        Column.DiffAEData.START_X,
        Column.DiffAEData.END_Y,
        Column.DiffAEData.END_X,
    ]

    bounding_box_columns = [
        Column.DiffAEData.START_Y,
        Column.DiffAEData.START_X,
        Column.DiffAEData.END_Y,
        Column.DiffAEData.END_X,
    ]

    global task

    def task(args: ImageProcessingArgs):
        # Load grid-based feature dataframe for selected dataset
        df_loc = get_dataframe_location_for_dataset(feature_manifest, args.dataset_name)
        df_ = load_dataframe(df_loc, delay=True)
        df = df_[columns_to_compute].compute()

        # Filter down to requested position and timepoint
        df = df[df[Column.POSITION] == args.position]
        df = df[df[Column.TIMEPOINT] == args.timepoint]

        # Load grid segmentation image
        dataset = load_dataset_config(args.dataset_name)
        seg_loc = get_image_location_for_dataset(image_manifest, dataset, position=args.position)
        seg = load_image(seg_loc, squeeze=True, compute=True, timepoints=args.timepoint)

        # Calculate region properties for segmentation image
        segprops = regionprops(label_image=seg)

        for prop in segprops:
            crop_index_from_seg = prop.label - 1
            crop_loc_matched = (
                df[df[Column.CROP_INDEX] == crop_index_from_seg][bounding_box_columns] == prop.bbox
            )

            if not all(crop_loc_matched):
                logger.error(
                    "Crop index '%d' in grid segmentation does not match bounding box for "
                    "position '%d' and timepoint '%d' in dataset '%s'",
                    crop_index_from_seg,
                    args.position,
                    args.timepoint,
                    args.dataset_name,
                )

    for dataset_name in dataset_names:
        progress_bar = ProgressBar([dataset_name], "Validating")
        progress_bar.set_iteration_name(dataset_name)

        analysis_queue = build_analysis_queue(
            dataset_names=[dataset_name],
            image_validation_frequency=None,
            t_start=0,
            t_final=10 if DEMO_MODE else None,
            max_positions=2 if DEMO_MODE else None,
        )

        process_task_queue(
            task,
            analysis_queue,
            description="Validating grid segmentation",
            num_processes=NUM_WORKERS or 1,
            chunksize=2,
        )

        progress_bar.set_step_description("Finished validation steps")
        progress_bar.update(1)
        progress_bar.close()


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
