"""
Custom type for patch type.

This module provides the ``PatchType`` type annotation, which is a wrapper
around a `str` that will:

- convert input to lower case
- check that the selected type is valid
- replace any '-' with '_'

## Example workflow usage

.. code-block:: python

    from endo_pipeline.cli import PatchType

    def main(patch_type: PatchType):
        print(patch_type)

## Example CLI usage

.. code-block:: bash

    # valid type (lower case)
    endopipe workflow --patch-type cell-centered

    # valid type (mixed case) will be converted to lower case
    endopipe workflow --patch-type Cell-Centered

    # invalid type will throw an error
    endopipe workflow --patch-type invalid
"""

from collections.abc import Sequence
from typing import Annotated, Literal

from cyclopts import Parameter, Token


def _patch_type_validator(_, values) -> None:
    """Validate patch type to confirm if it is a valid option."""

    if values not in ("cell_centered", "grid_based"):
        raise ValueError("Patch type must be 'cell_centered' or 'grid_based'.")


def _patch_type_converter(_, tokens: Sequence[Token]) -> str:
    """Convert CLI tokens into lowercase."""

    return tokens[0].value.lower().replace("-", "_")


PatchType = Annotated[
    Literal["cell_centered", "grid_based"],
    Parameter(
        converter=_patch_type_converter,  # run patch type converter on tokens
        validator=_patch_type_validator,  # run patch type validator on tokens
    ),
]
