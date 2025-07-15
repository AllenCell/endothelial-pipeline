"""Generic configuration I/O for Endo Pipeline."""

import logging
from pathlib import Path
from typing import Any

from mashumaro.codecs.yaml import YAMLDecoder

from .dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get path to config directory."""

    return Path(__file__).resolve().parents[1] / "configs"
