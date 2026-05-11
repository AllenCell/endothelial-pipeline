from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
    save_output: bool = True,
) -> None:
    """Run the CDH5-based measured features extraction workflow.

    #test-ready #cpu-only

    Measures cell segmentation alignment to flow, elongation, edge intensity, etc.

    To enter a list of datasets to analyze, use the following format:

    .. code-block:: bash

        --datasets 20250818_20X 20250618_20X
    **Workflow demo**

    The ``--demo-mode`` (``-d``) flag can be used to run a simplified version of
    this workflow for testing purposes (e.g. during code review). The workflow
    will only extract measured features from the first two positions and the
    first three timepoints for each of the given datasets.
    """
    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.shape_features import (
        build_cdh5_measured_features_tables_multiproc_wrapper,
        concatenate_and_save_feature_tables,
    )
    from endo_pipeline.library.process.general_image_preprocessing import (
        build_analysis_queue,
        process_task_queue,
    )

    logger = logging.getLogger(__name__)

    out_dir = get_output_path(__file__)

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")
    logger.info(f"datasets analyzed: {datasets}")

    analysis_queue = build_analysis_queue(
        datasets,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        image_validation_frequency=None,
        is_test=DEMO_MODE,
        t_start=0,
        t_final=3 if DEMO_MODE else None,
    )

    # measure features from CDH5 segmentations and save as a .parquet table
    process_task_queue(
        build_cdh5_measured_features_tables_multiproc_wrapper,
        analysis_queue,
        description="Getting cell features",
        num_processes=n_proc,
        chunksize=2,
    )

    # lastly, for each dataset concatenate the tables from each timepoint
    # into a single output table for that dataset
    if save_output:
        for dataset_name in datasets:
            concatenate_and_save_feature_tables(
                out_dir,
                dataset_name,
                out_file_suffix="cdh5_alignments",
                input_filename_contains="cdh5_alignments",
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )
            concatenate_and_save_feature_tables(
                out_dir,
                dataset_name,
                out_file_suffix="cdh5_segprops",
                input_filename_contains="cdh5_segprops",
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )

    logger.info("...done analysis!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
