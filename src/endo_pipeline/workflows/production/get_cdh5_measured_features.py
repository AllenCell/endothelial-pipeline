from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    num_processes: int = 1,
    save_output: bool = True,
) -> None:
    """
    Extract measure features based on CDH5 segmentations.

    #cdh5-segmentation #test-ready #cpu-only

    Features extracted from CDH5-based cell segmentation include alignment to
    flow, elongation, edge intensity, etc. Features are calculated for each
    timepoint and then combined into a single dataframe.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe get-cdh5-measured-features -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe get-cdh5-measured-features --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `live_cdh5_seg_based_feat_datasets` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will extract
    features for first three timepoints of the first two positions for the first
    dataset.

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

    out_dir = get_output_path("cdh5_measured_features")

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")

    analysis_queue = build_analysis_queue(
        datasets,
        out_dir=out_dir,
        image_validation_frequency=None,
        t_start=0,
        t_final=3 if DEMO_MODE else None,
        max_positions=2 if DEMO_MODE else None,
        overwrite=True,
        save_output=save_output,
    )

    logger.info("Starting feature extraction...")

    process_task_queue(
        build_cdh5_measured_features_tables_multiproc_wrapper,
        analysis_queue,
        description="Getting cell features",
        num_processes=num_processes,
        chunksize=2,
    )

    # Concatenate outputs into a single output table for each dataset
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

    logger.info("Finished extracting features!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
