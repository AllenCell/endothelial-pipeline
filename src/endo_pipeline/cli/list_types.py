"""
Custom CLI list type annotations.

This module provides the following type annotations:

- ``StrList`` wrapper around ``list[str]``
    - ``UniqueStrList`` also removes duplicates and sorts the list
- ``IntList`` wrapper around ``int[str]``
    - ``UniqueIntList`` also removes duplicates and sorts the list
- ``FloatList`` wrapper around ``float[str]``

These types are annotated to allow the parameter to consume multiple tokens and
remove the "empty" list option.

**Example workflow usage**

.. code-block:: python

    from endo_pipeline.cli import UniqueStrList

    def main(names: UniqueStrList) -> None:
        print(names)

**Example CLI usage**

.. code-block:: bash

    # provide list entries by position
    endopipe workflow a b c

    # provide list entries via multiple keywords
    endopipe workflow --names a --names b --names c

    # provide list entries via single keywords
    endopipe workflow --names a b c

    # remove duplicates and sort the list
    endopipe workflow b c c a
"""

from collections.abc import Sequence
from typing import Annotated

from cyclopts import Parameter, Token


def _unique_str_list_converter(_, tokens: Sequence[Token]) -> list[str]:
    """Convert CLI tokens into unique and sorted list of strings."""

    return sorted({str(token.value) for token in tokens})


def _unique_int_list_converter(_, tokens: Sequence[Token]) -> list[int]:
    """Convert CLI tokens into unique and sorted list of integers."""

    return sorted({int(token.value) for token in tokens})


StrList = Annotated[
    list[str],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
    ),
]

UniqueStrList = Annotated[
    list[str],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
        converter=_unique_str_list_converter,  # run unique string list converter on tokens
    ),
]

IntList = Annotated[
    list[int],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
    ),
]

UniqueIntList = Annotated[
    list[int],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
        converter=_unique_int_list_converter,  # run unique int list converter on tokens
    ),
]

FloatList = Annotated[
    list[float],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
    ),
]
