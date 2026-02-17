from .apps import pipeline_cli, workflow_cli
from .crop_pattern import CropPattern
from .datasets import Datasets
from .list_types import FloatList, IntList, OptionalFloatList, StrList, UniqueIntList, UniqueStrList

__all__ = [
    "CropPattern",
    "Datasets",
    "FloatList",
    "IntList",
    "OptionalFloatList",
    "StrList",
    "UniqueIntList",
    "UniqueStrList",
    "pipeline_cli",
    "workflow_cli",
]

DEMO_MODE = False
"""True if workflows should be run in demo mode, False otherwise."""

USE_STAGING = False
"""True to use staging environments, False otherwise."""

NUM_GPUS: int | None = None
"""Number of GPUs available to use. None if no GPUs are available."""
