"""Methods for detecting single timepoint outliers."""

import numpy as np

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.library.process.single_tp_outlier.bf_timepoint_outlier import detect_bf_outliers
from endo_pipeline.library.process.single_tp_outlier.gfp_timepoint_outlier import (
    detect_egfp_scope_errors,
)
from endo_pipeline.settings.column_names import ColumnNameType


def detect_single_timepoint_outliers(
    dataset_config: DatasetConfig, position: int, max_timepoints: int | None = None
) -> dict[ColumnNameType, int | list[int] | list[float] | np.ndarray]:
    """
    Detect single timepoint outlier for given dataset and position.


    Parameters
    ----------
    dataset_config
        Configuration object containing metadata and paths for the dataset.
    position
        The position index within the dataset to analyze.
    max_timepoints
        Maximum number of timepoints to use for detecting outliers.

    Returns
    -------
    :
        Dictionary containing detected outlier information.
    """

    outliers = detect_bf_outliers(dataset_config, position, max_timepoints)
    if dataset_config.duration > 1:
        outliers.update(detect_egfp_scope_errors(dataset_config, position, max_timepoints))

    return outliers
