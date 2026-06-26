from endo_pipeline.cli import UniqueStrList


def main(collections: UniqueStrList | None = None) -> None:
    from endo_pipeline.configs import (
        get_available_dataset_collection_names,
        get_unannotated_positions,
        load_dataset_collection_config,
        load_dataset_config,
    )

    collection_names = collections or get_available_dataset_collection_names()

    for collection_name in collection_names:
        collection_config = load_dataset_collection_config(collection_name)
        for dataset_name in collection_config.dataset_names:
            dataset_config = load_dataset_config(collection_name, dataset_name)
            positions_kept_for_analysis = get_unannotated_positions(dataset_config)

            print(
                f"Collection: {collection_name}, Dataset: {dataset_name}, "
                f"FOVs kept for analysis: {positions_kept_for_analysis}"
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
