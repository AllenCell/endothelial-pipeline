from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
    save_output: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Run the label-free nuclear prediction workflow on a dataset, list of datasets, or collection.

    #test-ready #gpu

    Label-free nuclei predictions are generated using a CellPose 3 model retrained from the
    default "nuclei" model.

    To enter a list of datasets to analyze, use the following format:

    .. code-block:: bash

        --datasets 20250818_20X 20250618_20X

    **Workflow demo**

    The ``--demo-mode`` (``-d``) flag can be used to run the workflow on the first timepoint
    of the first 2 positions for each of the given datasets for workflow testing purposes.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.general_image_preprocessing import (
        build_analysis_queue,
        process_task_queue,
    )
    from endo_pipeline.library.process.lib_generate_label_free_nuc_pred import (
        generate_labelfree_nuclei_predictions,
    )

    logger = logging.getLogger(__name__)

    out_dir = get_output_path(__file__)

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")
    logger.info(f"datasets to analyze: {datasets}")

    # Get a list of timepoints and associated arguments to process from the list
    # of datasets to analyze and create validation images every 48 timepoints (ie. 4hrs)
    analysis_queue = build_analysis_queue(
        datasets,
        out_dir=out_dir,
        save_output=save_output,
        overwrite=overwrite,
        image_validation_frequency=48,
        is_test=DEMO_MODE,
        t_start=0,
        t_final=1 if DEMO_MODE else None,
    )

    # Predict nuclei from brightfield images using the retrained CellPose model
    process_task_queue(
        generate_labelfree_nuclei_predictions,
        analysis_queue,
        num_processes=n_proc,
        description="Predicting nuclei",
        chunksize=5,
    )

    logger.info("...done analysis!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
