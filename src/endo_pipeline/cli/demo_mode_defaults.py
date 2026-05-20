import typing

if typing.TYPE_CHECKING:
    from endo_pipeline.cli import Datasets


def use_default_collection(datasets: "Datasets | None", collection_name: str) -> "Datasets":
    """
    Get datasets in default collection, if datasets are not provided.

    If running in demo mode, only return the first dataset in the collection.

    Parameters
    ----------
    datasets
        List of selected datasets.
    collection_name
        Name of collection if no datasets are provided.

    Returns
    -------
    :
        List of datasets.
    """

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection

    if datasets is None:
        datasets = get_datasets_in_collection(collection_name)

    if DEMO_MODE:
        datasets = datasets[:1]

    return datasets
