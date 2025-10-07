"""Methods for model config I/O."""

import logging
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


def get_model_config_dir() -> Path:
    """Get path to config directory."""

    return Path(__file__).resolve().parents[1] / "configs" / "models"


def load_model_config(model_config_name: str) -> "DictConfig | ListConfig":
    """Load single model config by name."""
    from omegaconf import OmegaConf

    config_dir = get_model_config_dir()
    config_file = config_dir / f"{model_config_name}.yaml"

    if not config_file.exists():
        logger.error("Model config [ %s ] could not be loaded", model_config_name)
        raise FileNotFoundError(f"No such file '{config_file}'")
    else:
        config = OmegaConf.load(config_file)
        return config
