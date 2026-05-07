from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    save_output: bool = True,
    n_proc: int = 1,
    concatenate_tables_only: bool = False,
) -> None:
    """Run workflow to measure features from label-free nuclei predictions.

    #test-ready #cpu-only

    Measures the label-free nuclei segmentation labels brightfield intensity and centroids and
    matches them to existing cell segmentation labels.

    To enter a list of datasets to analyze, use the following format:

    .. code-block:: bash

        --datasets 20250818_20X 20250618_20X

    **Workflow demo**

    The ``--demo-mode`` (``-d``) flag can be used to run the workflow on the first 3 timepoints
    of the first 2 positions for each of the given datasets for workflow testing purposes.
    """
    import logging

    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
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

    out_dir = get_output_path(__file__)

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")
    logger.info(f"datasets analyzed: {datasets}")

    if not concatenate_tables_only:
        # build analysis queue
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

        # get and save results from images in analysis queue
        process_task_queue(
            get_and_save_nuclei_features_arg_unpacker,
            analysis_queue,
            description="Getting nuclei features",
            num_processes=n_proc,
            chunksize=5,
        )

    # concatenate the results outputs from above in to a single table
    if save_output:
        for dataset_name in tqdm(
            datasets, desc="Replacing individual tables with combined table..."
        ):
            out_file_suffix = "nuclei_labelfree_features"

            concatenate_and_save_feature_tables(
                out_dir=out_dir,
                dataset_name=dataset_name,
                out_file_suffix=out_file_suffix,
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )

    logger.info("...done analysis!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
