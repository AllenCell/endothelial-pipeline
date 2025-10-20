from endo_pipeline.cli import Datasets

TAGS = ["quality-control", "preprocessing"]


def main(datasets: Datasets | None = None) -> None:
    """
    Detect and annotate outlier timepoints in BF and GFP channels.

    Parameters
    ----------
    datasets
        List of dataset names to process. If None, processes all datasets in the
        "live_20X_objective_3i_microscope" collection. If DEMO_MODE is enabled,
        only the first dataset will be processed.
    """

    import logging

    from endo_pipeline import DEMO_MODE
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
        datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

    for dataset_name in datasets:
        dataset_config = load_dataset_config(dataset_name)
        tp_annotations = (
            dataset_config.timepoint_annotations
            if dataset_config.timepoint_annotations is not None
            else {}
        )

        # Initialize annotations for each type
        tp_annotations[TimepointAnnotation.AUTO_BF_SCOPE_ERROR] = {
            position: [] for position in dataset_config.zarr_positions
        }
        tp_annotations[TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT] = {
            position: [] for position in dataset_config.zarr_positions
        }
        tp_annotations[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR] = {
            position: [] for position in dataset_config.zarr_positions
        }

        # Detect and annotate outliers for each position
        for position in dataset_config.zarr_positions:
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

        # Save the updated annotations back to the dataset configuration
        dataset_config.timepoint_annotations = tp_annotations
        save_dataset_config(dataset_config)

        if DEMO_MODE:
            logger.info(f"DEMO_MODE is ON. Processed only the first dataset: {dataset_name}")
            break


if __name__ == "__main__":

    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
