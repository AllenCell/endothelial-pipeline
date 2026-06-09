"""Data structures for dataframe manifests."""

from dataclasses import field
from pathlib import Path

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass


@dataclass
class DataframeLocation:
    """Storage locations for dataframes."""

    fmsid: str | None = None
    """FMS file id for dataframe."""

    path: Path | None = None
    """Local path to dataframe."""

    s3uri: str | None = None
    """S3 URI for dataframe (starting with s3://)."""


@dataclass
class DataframeManifest:
    """Dataframe manifest for pipeline."""

    name: str
    """Unique name of the dataframe manifest."""

    workflow: str
    """Name of workflow that produced the dataframes."""

    parameters: dict = field(default_factory=dict)
    """Specific workflow parameters used to produce the dataframes."""

    columns: dict[str, str] = field(default_factory=dict)
    """Column names and descriptions."""

    locations: dict[str, DataframeLocation] = field(default_factory=dict)
    """Locations of individual dataframes."""

    class Config(BaseConfig):
        """Settings for dataframe manifest."""

        forbid_extra_keys = True
        omit_none = False
