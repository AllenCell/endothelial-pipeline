def main() -> None:
    """
    Validate datasets in dataset collections.

    #validation #datasets
    """

    import logging

    from endo_pipeline.configs import (
        load_dataset_collection_config,
        load_dataset_config,
        validate_filtered_dataset_collection,
    )

    logger = logging.getLogger(__name__)

    logger.info("Starting validation of dataset collection configs")

    validate_filtered_dataset_collection("live", "20X", "3i")

    # validate 3d_flow_field_analysis collection
    flow_field_analysis_datasets = load_dataset_collection_config("3d_flow_field_analysis").datasets

    # confirm that they are all single flow condition datasets
    for dataset_name in flow_field_analysis_datasets:
        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.flow_conditions) != 1:
            logger.error(
                "Dataset [ %s ] in [ 3d_flow_field_analysis ] has multiple flow conditions.",
                dataset_name,
            )

    logger.info("Finished validation of dataset collection configs")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
