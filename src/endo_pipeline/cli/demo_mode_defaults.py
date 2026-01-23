import typing

if typing.TYPE_CHECKING:
    from endo_pipeline.cli import Datasets


def use_default_collection(datasets: "Datasets | None", collection_name: str) -> "Datasets":
    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection

    if datasets is None:
        datasets = get_datasets_in_collection(collection_name)
        if DEMO_MODE:
            # Only modify `datasets` if the user didn't specify them
            datasets = datasets[:1]
    return datasets
