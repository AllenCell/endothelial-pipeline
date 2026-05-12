from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
    save_output: bool = True,
) -> None:
    """Run the tracking workflow on a dataset, a list of datasets, or a dataset collection.
    Saves a table as a .parquet file containing the cell tracks for each dataset analyzed as well
    as their associated segmentation labels.

    #cdh5-segmentation #tracking #test-ready #cpu-only

    The tracking workflow loads the CDH5 segmentations from a single position in a single dataset
    and builds cell tracks by finding which cell segmentations at a given timepoint T and T+1 are
    most similar based on their % overlap. The segmentation and its match must be each other's best
    match based on % overlap, and if no such pairs are found between T and T+1 then the algorithm
    will search T+2, T+3, etc. up to a maximum of 4 timepoints ahead, allowing for a maximum
    of 3 missed timepoints (the "track tolerance") before terminating the track. New tracks are
    initiated for any segmentations at timepoint T that are not matched to any segmentations at
    timepoint T-1. The track tolerance is greater than 0 to introduce robustness to incorrect
    segmentations.


    To enter a list of datasets to analyze, use the following format:

    .. code-block:: bash

        --datasets 20250818_20X 20250618_20X

    **Workflow demo**

    The ``--demo-mode`` (``-d``) flag can be used to run the workflow on the first 10 timepoints
    of the first 2 positions for each of the given datasets for workflow testing purposes.
    """
    import logging

    import pandas as pd
    from tqdm import tqdm

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

    out_dir = get_output_path(__file__)

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")
    logger.info(f"datasets analyzed: {datasets}")

    analysis_queue = build_analysis_queue(
        datasets,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        is_test=DEMO_MODE,
        image_validation_frequency=None,
    )

    # Split analysis queue by dataset and position
    analysis_queue_df = pd.DataFrame(analysis_queue)
    analysis_queue_per_position = list(analysis_queue_df.groupby(["dataset_name", "position"]))

    # Run tracking algorithm on each position
    process_task_queue(
        run_tracking_multiproc_wrapper,
        analysis_queue_per_position,
        num_processes=n_proc,
        description="Tracking",
        chunksize=1,
    )

    # Combine tracking algorithm table output into one table per dataset
    if save_output:
        for dataset_name in tqdm(
            datasets, desc="Replacing individual tables with combined table..."
        ):
            concatenate_and_save_feature_tables(
                out_dir=out_dir,
                dataset_name=dataset_name,
                out_file_suffix="tracking",
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )

    logger.info("...done analysis!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
