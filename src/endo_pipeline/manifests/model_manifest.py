"""Data structures for model manifests."""

from dataclasses import field
from pathlib import Path

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass


@dataclass
class ModelLocation:
    """Storage locations for models."""

    mlflowid: str | None = None
    """MLFlow run id for model."""

    fmsid: str | tuple[str, str] | None = None
    """FMS file id for model."""

    path: Path | tuple[Path, Path] | None = None
    """Local path to model."""

    s3uri: str | tuple[str, str] | None = None
    """S3 URI for dataframe (starting with s3://)."""


@dataclass
class ModelManifest:
    """Model manifest for pipeline."""

    name: str
    """Unique name of the model manifest."""

    workflow: str
    """Name of workflow that produced the models."""

    parameters: dict = field(default_factory=dict)
    """Specific workflow parameters used to produce the models."""

    locations: dict[str, ModelLocation] = field(default_factory=dict)
    """Locations of individual models."""

    class Config(BaseConfig):
        """Settings for model manifest."""

        forbid_extra_keys = True
        omit_none = False
