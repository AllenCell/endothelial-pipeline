from endo_pipeline.cli import UniqueStrList


def main(collections: UniqueStrList | None = None) -> None:
    """
    Validate datasets in dataset collections.

    #datasets #validation

    For each dataset collection, confirm:

    - The dataset collection config exists and can be loaded
    - The dataset collection config follows the schema defined by `DatasetCollectionConfig`
    - All datasets in the collection have a corresponding config

    Certain dataset collections have additional validation steps:

    - `3d_flow_field_analysis`
        - All datasets must only have a single flow condition

    # - The flow field analysis datasets are all single flow condition datasets

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe validate-dataset-collections -vd
    ```

    To run the workflow for a single collection:

    ```bash
    uv run endopipe validate-dataset-collections --collections COLLECTION_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run
    validation on the first two collections.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_available_dataset_collection_names,
        get_available_dataset_names,
        get_datasets_in_collection,
        load_dataset_config,
        validate_dataset_collection_config,
    )
    from endo_pipeline.library.process.progress_bar import ProgressBar

    logger = logging.getLogger(__name__)

    collection_names = collections or get_available_dataset_collection_names()

    if DEMO_MODE:
        logger.warning("DEMO MODE - Only validating the first two collections")
        collection_names = collection_names[:2]

    all_dataset_names = get_available_dataset_names()

    for collection_name in collection_names:
        progress_bar = ProgressBar([collection_name], "Validating")
        progress_bar.set_iteration_name(collection_name)

        # Validate dataset collection config schema
        progress_bar.set_step_description("Checking dataset collection config matches schema")
        try:
            validate_dataset_collection_config(collection_name)
        except Exception:
            logger.error("Collection '%s' does not have valid schema", collection_name)
            progress_bar.set_step_description("Unable to finish validation steps")
            continue

        # Load dataset collection config
        dataset_names = get_datasets_in_collection(collection_name)

        # Check if all datasets in the collection exist
        progress_bar.set_step_description("Checking all datasets in collection have configs")
        for dataset_name in dataset_names:
            if dataset_name not in all_dataset_names:
                logger.error(
                    "Dataset '%s' in collection '%s' does not have matching config",
                    dataset_name,
                    collection_name,
                )

        # Specific validation for 3d_flow_field_analysis collection
        if collection_name == "3d_flow_field_analysis":
            for dataset_name in dataset_names:
                dataset_config = load_dataset_config(dataset_name)
                if len(dataset_config.flow_conditions) != 1:
                    logger.error(
                        "Dataset '%s' in collection '%s' has multiple flow conditions",
                        dataset_name,
                        collection_name,
                    )
                    continue

        progress_bar.set_step_description("Finished validation steps")
        progress_bar.update(1)
        progress_bar.close()


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
