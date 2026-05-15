from typing import Literal

from .apps import pipeline_cli, workflow_cli
from .crop_pattern import CropPattern
from .datasets import Datasets
from .list_types import FloatList, IntList, StrList, UniqueIntList, UniqueStrList

__all__ = [
    "CropPattern",
    "Datasets",
    "FloatList",
    "IntList",
    "StrList",
    "UniqueIntList",
    "UniqueStrList",
    "pipeline_cli",
    "workflow_cli",
]

DEMO_MODE = False
"""True if workflows should be run in demo mode, False otherwise."""

NUM_GPUS: int | None = None
"""Number of GPUs available to use. None if no GPUs are available."""

UPLOAD_TO_FMS = False
"""True to upload outputs to FMS, False otherwise."""

FMS_ENV: Literal["prod", "stg", "dev"] = "prod"
"""FMS environment to upload to."""
