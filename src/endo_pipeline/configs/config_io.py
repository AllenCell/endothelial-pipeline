"""Generic configuration I/O for Endo Pipeline."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get path to config directory."""

    return Path(__file__).resolve().parents[1] / "configs"
