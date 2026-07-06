from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None, overwrite: bool = True) -> None:
    """
    Run label-free nuclei prediction.

    #nuclei-prediction #test-ready #gpu #workers

    Label-free nuclei predictions are generated using a CellPose 3 model
    retrained from the default "nuclei" model. Validation images are saved every
    48 timepoints (i.e. 4 hrs).

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe run-labelfree-nuclei-prediction -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe run-labelfree-nuclei-prediction --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `live_cdh5_seg_based_feat_datasets` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will segment the
    first timepoint of the first two positions for the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to segment.
    overwrite
        True to overwrite existing images at output path, False otherwise.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE, NUM_WORKERS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.general_image_preprocessing import (
        build_analysis_queue,
        process_task_queue,
    )
    from endo_pipeline.library.process.lib_generate_label_free_nuc_pred import (
        generate_labelfree_nuclei_predictions,
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
        image_validation_frequency=48,
        t_start=0,
        t_final=1 if DEMO_MODE else None,
        max_positions=2 if DEMO_MODE else None,
        overwrite=overwrite,
    )

    logger.info("Starting nuclei prediction...")

    process_task_queue(
        generate_labelfree_nuclei_predictions,
        analysis_queue,
        num_processes=NUM_WORKERS or 1,
        description="Predicting nuclei",
        chunksize=5,
    )

    logger.info("Finished predicting nuclei!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
