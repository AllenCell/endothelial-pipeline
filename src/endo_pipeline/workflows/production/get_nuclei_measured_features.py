from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    num_processes: int = 1,
    save_output: bool = True,
) -> None:
    """
    Extract measure features based on label-free nuclei predictions.

    #nuclei-prediction #test-ready #cpu-only

    Measures the label-free nuclei segmentation labels brightfield intensity and
    centroids and matches them to existing cell segmentation labels.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe get-nuclei-measured-features -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe get-nuclei-measured-features --datasets DATASET_NAME
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
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.shape_features import (
        concatenate_and_save_feature_tables,
        get_and_save_nuclei_features_arg_unpacker,
    )
    from endo_pipeline.library.process.general_image_preprocessing import (
        build_analysis_queue,
        process_task_queue,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    analysis_queue = build_analysis_queue(
        dataset_names=dataset_names,
        out_dir=output_path,
        image_validation_frequency=None,
        t_start=0,
        t_final=3 if DEMO_MODE else None,
        max_positions=2 if DEMO_MODE else None,
        overwrite=True,
        save_output=save_output,
    )

    logger.info("Starting feature extraction...")

    process_task_queue(
        get_and_save_nuclei_features_arg_unpacker,
        analysis_queue,
        description="Getting nuclei features",
        num_processes=num_processes,
        chunksize=5,
    )

    # Concatenate outputs into a single output table for each dataset
    if save_output:
        for dataset_name in datasets:
            concatenate_and_save_feature_tables(
                output_path,
                dataset_name,
                out_file_suffix="nuclei_labelfree_features",
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )

    logger.info("Finished extracting features!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
