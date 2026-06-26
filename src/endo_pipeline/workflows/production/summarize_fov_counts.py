from endo_pipeline.cli import UniqueStrList


def main(collections: UniqueStrList | None = None) -> None:
    """
    Report the number of unannotated FOVs kept for analysis from the specified
    dataset collections.

    If no collections are specified, the default collections used in the paper
    will be reported (set via the COLLECTIONS_NAMED_IN_PAPER constant).

    """
    from endo_pipeline.configs import (
        get_unannotated_positions,
        load_dataset_collection_config,
        load_dataset_config,
    )

    COLLECTIONS_NAMED_IN_PAPER = [
        "diffae_model_training",
        "shear_stress",
        "perturbation",
        "nulcear_labelfree_model_training",
    ]

    collection_names = collections or COLLECTIONS_NAMED_IN_PAPER

    for collection_name in collection_names:
        collection_config = load_dataset_collection_config(collection_name)
        num_unannotated_positions = 0
        for dataset_name in collection_config.datasets:
            dataset_config = load_dataset_config(dataset_name)
            positions_kept_for_analysis = get_unannotated_positions(dataset_config)
            num_unannotated_positions += len(positions_kept_for_analysis)

        print(
            f"Collection: {collection_name}, "
            f"Num unannotated FOVs kept for analysis: {num_unannotated_positions}"
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
