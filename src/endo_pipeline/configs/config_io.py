"""Generic configuration I/O for Endo Pipeline."""

import logging
from pathlib import Path
from typing import Any

from mashumaro.codecs.yaml import YAMLDecoder

from src.endo_pipeline.configs import DatasetConfig

logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get path to config directory."""

    return Path(__file__).resolve().parents[1] / "configs"


def load_config(config_name: str = "diffae") -> dict[Any, Any]:
    """Load configuration for training DiffAE model from YAML file."""

    config_dir = get_config_dir()
    config_file = config_dir / f"{config_name}.yaml"

    if not config_file.exists():
        logger.error("Config [ %s ] could not be loaded", config_name)
        raise FileNotFoundError(f"No such file '{config_file}'")
    else:
        logger.debug("Loaded config[ %s ]", config_file)
        return YAMLDecoder(DatasetConfig).decode(config_file.read_text())
