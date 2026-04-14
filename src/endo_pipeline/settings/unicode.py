"""Global settings for unicode characters used in the project."""

from enum import StrEnum


class UnicodeCharacters(StrEnum):
    """Unicode characters used in the project."""

    THETA = "\u03b8"
    """Unicode for lowercase theta."""

    RHO = "\u03c1"
    """Unicode for lowercase rho."""

    MU = "\u03bc"
    """Unicode for lowercase mu."""

    SIGMA = "\u03c3"
    """Unicode for lowercase sigma."""

    RIGHT_ARROW = "\u2192"
    """Unicode for right arrow."""

    SQUARED = "\u00b2"
    """Unicode for superscript 2."""
