from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Detect and annotate outlier timepoints in BF and GFP channels.

    #quality-control #preprocessing #test-ready #cpu-only

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe annotate-timepoint-outliers -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe annotate-timepoint-outliers --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only annotate
    two positions of the first dataset, using the first 10 timepoints.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to annotate.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
        save_dataset_config,
    )
    from endo_pipeline.library.process.single_tp_outlier.bf_timepoint_outlier import (
        detect_bf_outliers,
    )
    from endo_pipeline.library.process.single_tp_outlier.gfp_timepoint_outlier import (
        detect_egfp_scope_errors,
    )

    logger = logging.getLogger(__name__)

    if datasets is None:
        datasets = get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.info("DEMO_MODE is ON. Processing only the first dataset.")
        datasets = datasets[:1]

    for dataset_name in datasets:
        dataset_config = load_dataset_config(dataset_name)

        positions = dataset_config.zarr_positions
        if DEMO_MODE:
            positions = positions[:1]
            logger.info(f"DEMO_MODE is ON. Processing only position: {positions}")

        tp_annotations = (
            dataset_config.timepoint_annotations
            if dataset_config.timepoint_annotations is not None
            else {}
        )

        # Initialize annotations for each type
        tp_annotations[TimepointAnnotation.AUTO_BF_SCOPE_ERROR] = {
            position: [] for position in positions
        }
        tp_annotations[TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT] = {
            position: [] for position in positions
        }
        tp_annotations[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR] = {
            position: [] for position in positions
        }

        # Detect and annotate outliers for each position
        for position in positions:
            bf_scope_error, bf_temp_artifact = detect_bf_outliers(
                dataset_config, position, visualize=True
            )
            tp_annotations[TimepointAnnotation.AUTO_BF_SCOPE_ERROR][position].extend(bf_scope_error)
            tp_annotations[TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT][position].extend(
                bf_temp_artifact
            )

            if dataset_config.duration > 1:
                egfp_scope_error = detect_egfp_scope_errors(
                    dataset_config, position, visualize=True
                )
                tp_annotations[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR][position].extend(
                    egfp_scope_error
                )

        if DEMO_MODE:
            logger.info("DEMO_MODE is ON. Skip saving annotation for single position.")
            continue

        # Save the updated annotations back to the dataset configuration
        dataset_config.timepoint_annotations = tp_annotations
        save_dataset_config(dataset_config)


if __name__ == "__main__":

    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
