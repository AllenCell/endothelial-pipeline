"""
Custom type for list of dataset names or dataset collection names.

This module provides the ``Datasets`` type annotation, which is a wrapper around
a `list[str]` that will:

- expand any valid dataset collection that are provided into dataset names
- check that all dataset names provides are valid (i.e. there is a corresponding
dataset config available)
- remove any duplicate datasets from the list of datasets

**Example workflow usage**

.. code-block:: python

    from endopipe.cli import Datasets
    from endo_pipeline.configs load_dataset_config

    def main(datasets: Datasets):
        dataset_configs = [load_dataset_config(dataset) for dataset in datasets]

**Example CLI usage**

.. code-block:: bash

    # load all provided datasets
    endopipe workflow --datasets dataset1 dataset2

    # load dataset collection
    endopipe workflow --datasets collection_name

    # load datasets and dataset collection
    endopipe workflow --datasets dataset1 dataset2 collection_name

    # load datasets and provide additional parameters
    endopipe workflow --datasets dataset1 dataset2 --param1 --param2

    # invalid dataset name or collection will throw an error
    endopipe workflow --datasets invalid
"""

from collections.abc import Sequence
from typing import Annotated

from cyclopts import Parameter, Token


def _dataset_validator(_, values) -> None:
    """Validate a list of dataset to confirm they are valid dataset names."""

    from endo_pipeline.configs import get_available_dataset_names

    if values is None:
        return

    available_dataset_names = get_available_dataset_names()

    for value in values:
        if value not in available_dataset_names:
            raise ValueError(f"[ {value} ] is not a valid dataset or dataset collection name.")


def _dataset_converter(_, tokens: Sequence[Token]) -> list[str]:
    """Convert CLI tokens into list of datasets by automatically dataset collections."""

    from endo_pipeline.configs import (
        get_available_dataset_collection_names,
        get_datasets_in_collection,
    )

    datasets = []

    available_dataset_collection_names = get_available_dataset_collection_names()

    for token in tokens:
        if token.value in available_dataset_collection_names:
            datasets.extend(get_datasets_in_collection(token.value))
        else:
            datasets.append(token.value)

    return list(set(datasets))


Datasets = Annotated[
    list[str],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
        converter=_dataset_converter,  # run dataset converter on tokens
        validator=_dataset_validator,  # run dataset validator on list of datasets
    ),
]
