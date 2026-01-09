"""
Custom CLI list type annotations.

This module provides the following type annotations:

- ``StringList`` wrapper around ``list[str]``
- ``IntList`` wrapper around ``int[str]``
- ``FloatList`` wrapper around ``float[str]``

These types are annotated to allow the parameter to consume multiple tokens and
remove the "empty" list option.

**Example workflow usage**

.. code-block:: python

    from endo_pipeline.cli import StringList

    def main(names: StringList) -> None:
        print(names)

**Example CLI usage**

.. code-block:: bash

    # provide list entries by position
    endopipe workflow a b c

    # provide list entries via multiple keywords
    endopipe workflow --names a --names b --names c

    # provide list entries via single keywords
    endopipe workflow --names a b c
"""

from typing import Annotated

from cyclopts import Parameter

StringList = Annotated[
    list[str],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
    ),
]

IntList = Annotated[
    list[int],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
    ),
]

FloatList = Annotated[
    list[float],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
    ),
]
