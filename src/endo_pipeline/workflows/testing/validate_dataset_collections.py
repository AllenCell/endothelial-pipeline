def main() -> None:
    """
    Validate datasets in dataset collections.

    #datasets #validation

    Dataset validations include:

    - The live 20x 3i dataset collection contains all matching datasets
    - The flow field analysis datasets are all single flow condition datasets

    ## Example usage

    To run the full workflow:

    ```bash
    uv run endopipe validate-dataset-collections
    ```
    """

    import logging

    from endo_pipeline.configs import (
        load_dataset_collection_config,
        load_dataset_config,
        validate_filtered_dataset_collection,
    )

    logger = logging.getLogger(__name__)

    print("Validating live 20x 3i dataset collection")
    validate_filtered_dataset_collection("live", "20X", "3i")

    print("Validating flow field analysis datasets")
    flow_field_analysis_datasets = load_dataset_collection_config("3d_flow_field_analysis").datasets

    # confirm that they are all single flow condition datasets
    for dataset_name in flow_field_analysis_datasets:
        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.flow_conditions) != 1:
            logger.error(
                "Dataset [ %s ] in [ 3d_flow_field_analysis ] has multiple flow conditions.",
                dataset_name,
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
