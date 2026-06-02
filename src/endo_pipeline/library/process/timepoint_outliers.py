"""Methods for detecting single timepoint outliers."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.io import load_image
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.image_data import NUM_ZSLICES
from endo_pipeline.settings.method_constants import (
    BF_ROLLING_WINDOW,
    GFP_ROLLING_WINDOW,
    OUTLIER_THRESHOLD,
    PARTIAL_DARK_THRESHOLD,
)


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

    outliers = detect_single_timepoint_bf_outliers(dataset_config, position, max_timepoints)
    if dataset_config.duration > 1:
        outliers.update(
            detect_single_timepoint_gfp_outliers(dataset_config, position, max_timepoints)
        )

    return outliers


def detect_single_timepoint_bf_outliers(
    dataset_config: DatasetConfig,
    position: int,
    max_timepoints: int | None = None,
    rolling_window: int = BF_ROLLING_WINDOW,
    outlier_threshold: float = OUTLIER_THRESHOLD,
    partial_threshold: float = PARTIAL_DARK_THRESHOLD,
) -> dict[ColumnNameType, int | list[int] | list[float] | np.ndarray]:
    """
    Detect outliers in brightfield (BF) microscopy data based on intensity
    thresholds.

    Parameters
    ----------
    dataset_config
        Configuration object containing metadata and paths for the dataset.
    position
        The position index within the dataset to analyze.
    max_timepoints
        Maximum number of timepoints to evaluate.
    rolling_window
        Size of the rolling window used for calculating the rolling mean.
    outlier_threshold
        Percentage to use for thresholding dark and bright BF outliers.
    partial_threshold
        Percentage to use for thresholding partial dark BF outliers.

    Returns
    -------
    :
        Dictionary of BF outlier detection results.
    """

    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    bf_zarr = load_image(zarr_loc, channels=["BF"], level=1, squeeze=True)

    # Compute mean intensity over x/y axes
    intensity_array = bf_zarr.mean(axis=(-2, -1))
    if max_timepoints is not None:
        intensity_array = intensity_array[:max_timepoints, :]
    flattened_img_data = intensity_array.flatten()

    # Convert to pandas Series for rolling median
    data_np = flattened_img_data.compute()
    series = pd.Series(data_np)
    rolling_median = series.rolling(rolling_window, center=True).median()

    # Pad edges
    start_pad_value = np.median(data_np[:rolling_window])
    end_pad_value = np.median(data_np[-rolling_window:])
    rolling_median.iloc[: rolling_window // 2] = start_pad_value
    rolling_median.iloc[-rolling_window // 2 :] = end_pad_value
    rolling_median_np = rolling_median.to_numpy()

    # Thresholds
    dark_threshold = rolling_median_np * (1 - outlier_threshold)
    partial_dark_threshold = rolling_median_np * (1 - partial_threshold)
    bright_threshold = rolling_median_np * (1 + outlier_threshold)

    # Peaks
    minima, _ = find_peaks(-data_np)  # dark
    maxima, _ = find_peaks(data_np)  # bright

    # Outlier classification
    dark_outliers = [i for i in minima if data_np[i] <= dark_threshold[i]]
    partial_dark_outliers = [
        i for i in minima if data_np[i] <= partial_dark_threshold[i] and i not in dark_outliers
    ]
    bright_outliers = [i for i in maxima if data_np[i] >= bright_threshold[i]]

    bf_scope_error = sorted({int(idx // NUM_ZSLICES) for idx in partial_dark_outliers})
    bf_temp_artifact = sorted(
        {int(idx // NUM_ZSLICES) for idx in (dark_outliers + bright_outliers)}
    )

    return {
        Column.POSITION: position,
        Column.Annotations.BF_MEAN_INTENSITY: data_np,
        Column.Annotations.BF_ROLLING_MEDIAN: rolling_median_np,
        Column.Annotations.BF_DARK_THRESHOLD: dark_threshold,
        Column.Annotations.BF_PARTIAL_DARK_THRESHOLD: partial_dark_threshold,
        Column.Annotations.BF_BRIGHT_THRESHOLD: bright_threshold,
        Column.Annotations.BF_DARK_OUTLIERS: dark_outliers,
        Column.Annotations.BF_PARTIAL_DARK_OUTLIERS: partial_dark_outliers,
        Column.Annotations.BF_BRIGHT_OUTLIERS: bright_outliers,
        Column.Annotations.AUTO_BF_SCOPE_ERROR: bf_scope_error,
        Column.Annotations.AUTO_BF_TEMP_ARTIFACT: bf_temp_artifact,
    }


def detect_single_timepoint_gfp_outliers(
    dataset_config: DatasetConfig,
    position: int,
    max_timepoints: int | None = None,
    rolling_window: int = GFP_ROLLING_WINDOW,
    outlier_threshold: float = OUTLIER_THRESHOLD,
) -> dict[ColumnNameType, int | list[int] | list[float] | np.ndarray]:
    """
    Detect EGFP scope errors based on per-timepoint mean with rolling mean ±
    percentage thresholds.

    This function computes the mean intensity for each timepoint using all
    z-slices and identifies outlier timepoints based on a rolling mean and
    percentage thresholds. Optionally, it can visualize the results.

    Parameters
    ----------
    dataset_config
        Configuration object containing metadata and paths for the dataset.
    position
        The position index within the dataset to analyze.
    max_timepoints
        Maximum number of timepoints to evaluate.
    window
        Size of the rolling window used for calculating the rolling mean.
    percent
        Threshold percentage for identifying outliers.

    Returns
    -------
    :
        Dictionary of EGFR scope error detection results.
    """

    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    gfp_zarr = load_image(zarr_loc, channels=["EGFP"], level=1, squeeze=True)

    # Compute mean intensity across spatial dimensions (Y, X)
    intensity_array = gfp_zarr.mean(axis=(-2, -1))  # now (T, Z)
    if max_timepoints is not None:
        intensity_array = intensity_array[:max_timepoints, :]

    # Compute per-timepoint mean (across Z)
    tp_means = intensity_array.mean(axis=1).compute().astype(float)  # shape (T,)

    # Rolling median
    series = pd.Series(tp_means)
    rolling_median = series.rolling(rolling_window, center=True).median()

    # Pad edges
    start_val = np.nanmedian(tp_means[:rolling_window])
    end_val = np.nanmedian(tp_means[-rolling_window:])
    rolling_median.iloc[: rolling_window // 2] = start_val
    rolling_median.iloc[-rolling_window // 2 :] = end_val
    rolling_median = rolling_median.to_numpy()

    # Thresholds
    lower_threshold = rolling_median * (1 - outlier_threshold)
    upper_threshold = rolling_median * (1 + outlier_threshold)

    # Outlier timepoints
    dark_outliers = np.where(tp_means < lower_threshold)[0].tolist()
    bright_outliers = np.where(tp_means > upper_threshold)[0].tolist()

    egfp_scope_error = sorted(set(dark_outliers + bright_outliers))

    return {
        Column.POSITION: position,
        Column.Annotations.GFP_TIMEPOINT_MEANS: tp_means,
        Column.Annotations.GFP_ROLLING_MEDIAN: rolling_median,
        Column.Annotations.GFP_LOWER_THRESHOLD: lower_threshold,
        Column.Annotations.GFP_UPPER_THRESHOLD: upper_threshold,
        Column.Annotations.GFP_DARK_OUTLIERS: dark_outliers,
        Column.Annotations.GFP_BRIGHT_OUTLIERS: bright_outliers,
        Column.Annotations.AUTO_GFP_SCOPE_ERROR: egfp_scope_error,
    }
