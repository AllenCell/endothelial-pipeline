from endo_pipeline.cli import UniqueStrList


def main(collections: UniqueStrList | None = None) -> None:
    """
    Summarize number of unannotated FOVs kept for analysis.

    #datasets

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe summarize-fov-counts -vd
    ```

    To run the workflow for a single collection:

    ```bash
    uv run endopipe summarize-fov-counts --collections COLLECTION_NAME
    ```

    ## Dataset collection

    If collections are not provided, the workflow will use all the collections
    from the paper.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will summarize the
    first collection.

    Parameters
    ----------
    collections
        List of dataset collections to summarize.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_unannotated_positions,
        load_dataset_collection_config,
        load_dataset_config,
    )

    logger = logging.getLogger(__name__)

    COLLECTIONS_NAMED_IN_PAPER = [
        "diffae_model_training",
        "shear_stress",
        "perturbation",
        "nuclear_labelfree_model_training",
    ]

    collection_names = collections or COLLECTIONS_NAMED_IN_PAPER

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one collection")
        collection_names = collection_names[:1]

    for collection_name in collection_names:
        collection_config = load_dataset_collection_config(collection_name)
        num_unannotated_positions = 0

        for dataset_name in collection_config.datasets:
            dataset_config = load_dataset_config(dataset_name)
            positions_kept_for_analysis = get_unannotated_positions(dataset_config)
            num_unannotated_positions += len(positions_kept_for_analysis)

        print(
            f"Collection: {collection_name}, "
            f"Number of unannotated FOVs kept for analysis: {num_unannotated_positions}"
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
