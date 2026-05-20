from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    num_processes: int = 1,
    save_output: bool = True,
) -> None:
    """
    Run tracking on CDH5 class segmentations.

    #cdh5-segmentation #tracking #test-ready #cpu-only

    The workflow loads the CDH5 segmentations from a single position in a single
    dataset and builds cell tracks by finding which cell segmentations at a
    given timepoint T and T+1 are most similar based on their % overlap. The
    segmentation and its match must be each other's best match based on %
    overlap, and if no such pairs are found between T and T+1 then the algorithm
    will search T+2, T+3, etc. up to a maximum of 4 timepoints ahead, allowing
    for a maximum of 3 missed timepoints (the "track tolerance") before
    terminating the track. New tracks are initiated for any segmentations at
    timepoint T that are not matched to any segmentations at timepoint T-1. The
    track tolerance is greater than 0 to introduce robustness to incorrect
    segmentations.

    Outputs are saved as a .parquet file containing the cell tracks for each
    dataset analyzed as well as their associated segmentation labels.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe run-cdh5-tracking -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe run-cdh5-tracking --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `live_cdh5_seg_based_feat_datasets` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will segment the
    first 10 timepoints of the first two positions for the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to segment.
    num_processes
        Number of processes to use.
    save_output
        True to save outputs from workflow, False otherwise.
    """

    import logging
    from itertools import groupby

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.shape_features import concatenate_and_save_feature_tables
    from endo_pipeline.library.process.general_image_preprocessing import (
        build_analysis_queue,
        process_task_queue,
    )
    from endo_pipeline.library.process.lib_tracking import run_tracking_multiproc_wrapper

    logger = logging.getLogger(__name__)

    out_dir = get_output_path("cdh5_tracking")

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")

    analysis_queue = build_analysis_queue(
        datasets,
        out_dir=out_dir,
        image_validation_frequency=None,
        t_start=0,
        t_final=10 if DEMO_MODE else None,
        max_positions=2 if DEMO_MODE else None,
    )

    logger.info("Starting tracking...")

    # Group analysis queue by dataset and position
    analysis_queue_per_position = []
    for key, group in groupby(analysis_queue, lambda x: (x.dataset_name, x.position, x.output_dir)):
        analysis_queue_per_position.append([key, list(group)])

    process_task_queue(
        run_tracking_multiproc_wrapper,
        analysis_queue_per_position,
        num_processes=num_processes,
        description="Tracking",
        chunksize=1,
    )

    # Concatenate outputs into a single output table for each dataset
    if save_output:
        for dataset_name in datasets:
            concatenate_and_save_feature_tables(
                out_dir=out_dir,
                dataset_name=dataset_name,
                out_file_suffix="tracking",
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )

    logger.info("Finished tracking!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
