"""
Custom type for crop pattern.

This module provides the ``CropPattern`` type annotation, which is a wrapper
around a `str` that will:

- convert input to lower case
- check that the selected pattern is valid

## Example workflow usage

.. code-block:: python

    from endo_pipeline.cli import CropPattern

    def main(crop_pattern: CropPattern):
        print(crop_pattern)

**Example CLI usage**

.. code-block:: bash

    # valid pattern (lower case)
    endopipe workflow --crop-pattern grid

    # valid pattern (mixed case) will be converted to lower case
    endopipe workflow --crop-pattern Grid

    # invalid dataset name or collection will throw an error
    endopipe workflow --crop-pattern invalid
"""

from collections.abc import Sequence
from typing import Annotated, Literal

from cyclopts import Parameter, Token


def _crop_pattern_validator(_, values) -> None:
    """Validate crop pattern to confirm if it is a valid option."""

    if values not in ("grid", "tracked"):
        raise ValueError("Crop pattern must be 'grid' or 'tracked'.")


def _crop_pattern_converter(_, tokens: Sequence[Token]) -> str:
    """Convert CLI tokens into lowercase."""

    return tokens[0].value.lower()


CropPattern = Annotated[
    Literal["grid", "tracked"],
    Parameter(
        converter=_crop_pattern_converter,  # run crop pattern converter on tokens
        validator=_crop_pattern_validator,  # run crop pattern validator on tokens
    ),
]
