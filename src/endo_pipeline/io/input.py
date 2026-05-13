"""Methods for loading inputs."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_repository_root_dir() -> Path:
    """
    Get path to root of git repository.

    Returns
    -------
    :
        Path object for root of git repository.
    """

    return Path(__file__).resolve().parents[3]
