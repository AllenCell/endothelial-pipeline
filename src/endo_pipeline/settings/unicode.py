"""Global settings for unicode characters used in the project."""

from enum import StrEnum


class UnicodeCharacters(StrEnum):
    """Unicode characters used in the project."""

    ALPHA = "\u03b1"
    """Unicode for lowercase alpha."""

    MU = "\u03bc"
    """Unicode for lowercase mu."""

    PI = "\u03c0"
    """Unicode for lowercase pi."""

    RHO = "\u03c1"
    """Unicode for lowercase rho."""

    SIGMA = "\u03c3"
    """Unicode for lowercase sigma."""

    THETA = "\u03b8"
    """Unicode for lowercase theta."""

    DELTA = "\u0394"
    """Unicode for uppercase delta."""

    GEQ = "\u2265"
    """Unicode for greater than or equal to (using LaTeX naming convention)."""

    RIGHT_ARROW = "\u2192"
    """Unicode for right arrow."""

    SQUARED = "\u00b2"
    """Unicode for superscript 2."""

    PLUS_MINUS = "\u00b1"
    """Unicode for plus-minus sign."""

    MINUS = "\u2212"
    """Unicode for minus sign (using LaTeX naming convention)."""

    O_WITH_CIRCUMFLEX = "\u00f4"
    """Unicode for lowercase o with circumflex (e.g. It\u00f4)."""

    CHI = "\u03c7"
    """Unicode for lowercase chi."""

    R_SUBSCRIPT = "\u1d63"
    """Unicode for subscript r."""
