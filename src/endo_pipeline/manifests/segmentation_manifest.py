"""Data structures for segmentation manifests."""

from dataclasses import field
from pathlib import Path

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass


@dataclass
class SegmentationLocation:
    """Storage locations for segmentations."""

    path: Path | None = None
    """Local path to segmentation."""


@dataclass
class SegmentationManifest:
    """Segmentation manifest for pipeline."""

    name: str
    """Unique name of the segmentation manifest."""

    workflow: str
    """Name of workflow that produced the segmentations."""

    parameters: dict = field(default_factory=dict)
    """Specific workflow parameters used to produce the segmentations."""

    locations: dict[str, SegmentationLocation] = field(default_factory=dict)
    """Locations of individual segmentations."""

    class Config(BaseConfig):
        """Settings for segmentation manifest."""

        forbid_extra_keys = True
        omit_none = False
