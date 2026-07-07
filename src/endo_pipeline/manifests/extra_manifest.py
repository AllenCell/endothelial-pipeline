"""Data structures for extra manifests."""

from dataclasses import field
from pathlib import Path

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass


@dataclass
class ExtraLocation:
    """Storage locations for extras."""

    path: Path | None = None
    """Local path to extra."""

    s3uri: str | None = None
    """S3 URI for extra (starting with s3://)."""


@dataclass
class ExtraManifest:
    """Extra manifest for pipeline."""

    name: str
    """Unique name of the extra manifest."""

    workflow: str
    """Name of workflow that produced the extras."""

    parameters: dict = field(default_factory=dict)
    """Specific workflow parameters used to produce the extras."""

    locations: dict[str, ExtraLocation] = field(default_factory=dict)
    """Locations of individual extras."""

    class Config(BaseConfig):
        """Settings for extra manifest."""

        forbid_extra_keys = True
        omit_none = False
