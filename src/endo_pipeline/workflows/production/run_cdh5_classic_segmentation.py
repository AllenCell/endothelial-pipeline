from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    num_processes: int = 1,
) -> None:
    """
    Run CDH5 classic segmentation.

    #cdh5-segmentation #test-ready #cpu-only

    A summary of the segmentation process is as follows:

    1. load a Z-stack of the raw CDH5 signal for 1 timepoint from the timelapse
    2. create a maximum intensity projection (MIP) of the CDH5 Z-stack
    3. preprocess the MIP (gaussian filtering, rescaling, then rolling ball
       background subtraction)
    4. apply hysteresis thresholding on the processed MIP to estimate the CDH5
       signal and separate small round objects from long noodly objects
    5. create an initial cell segmentation from the CDH5 signal by
        a. producing a bunch of cell segmentation fragments using the watershed
           algorithm by using a distance transform to produce seeds for
           watershed, the CDH5 hysteresis threshold to limit watershed basin
           growth, and the preprocessed MIP as the image for watershed
        b. converting the cell segmentation fragments to a region adjacency
           graph (RAG) and merging fragments based on if the CDH5 signal
           intensity along their shared borders is too weak
    6. load the label-free nuclei predictions for the same timepoint
    7. use the nuclei predictions to split up regions that were incorrectly
       merged during the RAG merging step above by re-running a watershed
       segmentation using nuclei as seeds for any multinucleate regions and
       skeletonizations of the existing RAG-merged segmentations for any regions
       with one or no nuclei to produce an augmented / refined segmentation
    8. save the refined CDH5-based cell segmentation as a labeled image
    9. save multichannel validation images showing the raw CDH5 signal, the
       preprocessed CDH5, the  watershed seeds, the initial piece-wise
       segmentations, RAG-merged segmentations, and refined segmentations with
       their borders every 48th timepoint

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe run-cdh5-classic-segmentation -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe run-cdh5-classic-segmentation --datasets DATASET_NAME
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
    num_processes
        Number of processes to use.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.cdh5_preprocessing import (
        generate_cdh5_segmentation_refined_multiproc_wrapper,
    )
    from endo_pipeline.library.process.general_image_preprocessing import (
        build_analysis_queue,
        process_task_queue,
    )

    logger = logging.getLogger(__name__)

    out_dir = get_output_path("cdh5_classic_seg")

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")

    analysis_queue = build_analysis_queue(
        datasets,
        out_dir=out_dir,
        image_validation_frequency=48,
        t_start=0,
        t_final=1 if DEMO_MODE else None,
        max_positions=2 if DEMO_MODE else None,
        overwrite=True,
    )

    logger.info("Starting image segmentation...")

    process_task_queue(
        generate_cdh5_segmentation_refined_multiproc_wrapper,
        analysis_queue,
        num_processes=num_processes,
        description="Segmenting",
        chunksize=5,
    )

    logger.info("Finished segmenting images!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
