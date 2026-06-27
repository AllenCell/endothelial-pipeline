from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None, num_processes: int = 1) -> None:
    """
    Merge all cell-centered segmentation and PCA-reduced DiffAE features.

    #cdh5-segmentation #cdh5-tracking #nuclei-prediction #diffae #cell-centered

    This workflow combines the merged segmentation features from the
    `merge-segmentation-feature-tables` workflow (which itself merges features
    from the `get-cdh5-measured-features`, `get-nuclei-measured-features` and
    `run-cdh5-tracking` workflows) with the PCA-reduced DiffAE features produced
    by applying the `calculate-pca-features` workflow on tracked crops.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe combine-cell-centered-features -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe combine-cell-centered-features --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `live_cdh5_seg_based_feat_datasets` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will combine
    features for only the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to combine.
    num_processes
        Number of processes to use.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.track_integration import (
        get_and_save_pc_diffae_feats_liveseg_feats_merged_table_wrapper,
    )
    from endo_pipeline.library.process.general_image_preprocessing import process_task_queue

    logger = logging.getLogger(__name__)

    out_dir = get_output_path("cell_centered_features")

    datasets = datasets or get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        datasets = datasets[:1]

    analysis_queue = [(dataset, out_dir) for dataset in datasets]

    logger.info("Starting combining features...")

    process_task_queue(
        get_and_save_pc_diffae_feats_liveseg_feats_merged_table_wrapper,
        analysis_queue,
        description="Combining features",
        num_processes=num_processes,
        chunksize=1,
    )

    logger.info("Finished combining features!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
