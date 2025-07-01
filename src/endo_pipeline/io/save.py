"""Methods for saving outputs."""

import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_output_path(workflow_name: str, *subdirs: str, include_timestamp: bool = True) -> Path:
    """
    Create output directory for given workflow.

    Examples
    --------
    >>> get_output_path(__file__)
    Path("/path/to/results/2025-07-01/workflow_name")

    >>> get_output_path(__file__, subdir1, subdir2)
    Path("/path/to/results/workflow_name/subdir1/subdir2")

    >>> get_output_path(__file__, include_timestamp=False)
    Path("/path/to/results/workflow_name")

    Parameters
    ----------
    workflow_name
        Workflow name, directly specified or given by passing `__file__`.
    subdirs
        Zero or more additional subdirectories to include in file path.
    include_timestamp
        True to include YYYY-MM-DD timestamp in file path, False otherwise.

    Returns
    -------
    :
        Path object for output
    """

    output_dir = Path(__file__).resolve().parents[3] / "results"

    if include_timestamp:
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
        output_path = Path(output_dir, timestamp, Path(workflow_name).stem, *subdirs)
    else:
        output_path = Path(output_dir, Path(workflow_name).stem, *subdirs)

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directory [ %s ]", output_path)

    return output_path
