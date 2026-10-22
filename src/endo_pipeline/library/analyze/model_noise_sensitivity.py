"""Methods for analyzing the sensitivity of ML models to noise in input images."""

import logging

import numpy as np
import pandas as pd

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    load_dataset_config,
)
from endo_pipeline.library.process.image_processing import load_processed_bf_std_dev_image
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL

logger = logging.getLogger(__name__)


def get_bf_frame_intensities_multiproc_wrapper(args):
    dataset_name = args.dataset_name
    position = args.position
    timepoint = args.timepoint

    return get_bf_frame_intensities(dataset_name, position, timepoint)


def get_bf_frame_intensities(
    dataset_name: str,
    position: int,
    timepoint: int,
) -> dict:
    """
    Get mean and standard deviation of the log-normalized brightfield standard deviation projection
    along the Z-axis per frame with brightfield artifacts excluded.

    The timepoint outlier annotations store the mean BF intensity over XY for
    each z-slice of each timepoint, flattened into a single array per position.
    Intensities are averaged over z-slices to get a single intensity per frame.

    Parameters
    ----------
    dataset_name
        Name of the dataset.
    position
        Position within the dataset.
    timepoint
        Timepoint within the position.

    Returns
    -------
    :
        Mean BF intensity per position and timepoint.
    """

    bf_annotations = [
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
    ]

    dataset_config = load_dataset_config(dataset_name)

    excluded_timepoints = get_annotated_timepoints_for_position(
        dataset_config, position, bf_annotations
    )

    # If the timepoint is one with artifacts then exit early
    if timepoint in excluded_timepoints:
        return {}  # Return empty dict if timepoint is excluded

    image_delayed = load_processed_bf_std_dev_image(
        dataset_config, position, timepoint, DIFFAE_ZARR_RESOLUTION_LEVEL
    )
    image = image_delayed.compute()

    # Reduce over the last three axes (C, Y, X) to get per-frame statistics
    measurements = {
        Column.DATASET: dataset_name,
        Column.POSITION: position,
        Column.TIMEPOINT: timepoint,
        Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MEAN: image.mean(),
        Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_STD: image.std(),
        Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MEDIAN: np.median(image),
        Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MAX: image.max(),
        Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MIN: image.min(),
        Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_VARIANCE: image.var(),
    }

    return measurements


def summarize_bf_frame_intensities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize mean brightfield (BF) frame intensities for each dataset.

    Summary includes the number of positions, descriptive statistics, and the
    coefficient of variation (CV) of mean BF intensity across all frames.

    Parameters
    ----------
    df
        Dataframe of mean BF intensity per frame for one or more datasets.

    Returns
    -------
    :
        Dataframe of summary statistics per dataset.
    """

    grouped = df.groupby(Column.DATASET, sort=False)

    summary = grouped[Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MEAN].describe()

    # Rename summary columns to the standard ones in the code base
    summary = summary.rename(
        columns={
            "mean": Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MEAN,
            "50%": Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MEDIAN,
            "std": Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_STD,
            "max": Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MAX,
            "min": Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MIN,
        }
    )
    return summary.reset_index()
