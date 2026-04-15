"""Feature metadata structure and mapping to column names."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from endo_pipeline.io import slugify

MIN_VALUE = Literal["min"]
"""Use minimum value from data for feature limits."""

MAX_VALUE = Literal["max"]
"""Use maximum value from data for feature limits."""


class FeatureType(StrEnum):
    """Feature type."""

    CONTINUOUS = "continuous"
    """Feature has continuous values."""

    DISCRETE = "discrete"
    """Feature only has discrete values."""

    BOOLEAN = "boolean"
    """Feature is boolean."""


@dataclass
class FeatureMetadata:
    """Feature metadata."""

    name: str
    """Full feature name in title case."""

    label: str | None = None
    """Short feature label in title case. If not provided, set equal to name."""

    unit: str | None = None
    """Unit of the feature."""

    description: str | None = None
    """Description of the feature."""

    min: float | MIN_VALUE | None = None
    """Minimum value for feature."""

    max: float | MAX_VALUE | None = None
    """Maximum value for feature."""

    type: FeatureType = FeatureType.CONTINUOUS
    """Feature type."""

    bin_width: float | None = None
    """Width of bins."""

    ticks: range | None = None
    """Range for ticks."""

    slug: str = field(init=False)
    """Slug version of name."""

    name_with_unit: str = field(init=False)
    """Feature name with unit appended."""

    label_with_unit: str = field(init=False)
    """Feature label with unit appended."""

    limits: tuple[float | None | MIN_VALUE, float | None | MAX_VALUE] = field(init=False)
    """Minimum and maximum values of the feature as a tuple."""

    def __post_init__(self):
        """Post initialization steps for feature metadata."""

        # If label is not provided, set equal to the name.
        if self.label is None:
            self.label = self.name

        # Create versions of the name and label with unit.
        unit = f" ({self.unit})" if self.unit else ""
        self.name_with_unit = f"{self.name}{unit}"
        self.label_with_unit = f"{self.label}{unit}"

        # Create slug version of the name (useful for saving to files).
        self.slug = slugify(self.name_with_unit)

        # Set limits using minimum and maximum.
        self.limits = (self.min, self.max)
