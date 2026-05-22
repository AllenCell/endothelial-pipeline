"""Methods related to computing long-time-scale statistics of time series data."""

import numpy as np
import pandas as pd

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_track_length,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    get_kernel_density_estimate_from_histogram,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KernelName, KramersMoyalKernel
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_track_duration_to_dataframe,
)
from endo_pipeline.settings.column_names import ColumnName as Column


def compute_kde_on_bins(
    data: np.ndarray,
    bins: np.ndarray,
    kernel_name: KernelName,
    kernel_bandwidth: float,
    kernel_period: float | None,
) -> np.ndarray:
    """
    Compute a kernel density estimate (KDE) on the native histogram bin centers.

    Parameters
    ----------
    data
        1D array of data points to estimate the density for.
    bins
        Pre-computed bin edges (1D array) to use for the histogram and KDE.
    kernel_name
        The name of the kernel to use for the KDE.
    kernel_bandwidth
        The bandwidth parameter for the kernel density estimate.
    kernel_period
        The period for periodic kernels (pass None for non-periodic kernels).

    Returns
    -------
    :
        Kernel density estimate (KDE) values corresponding to the centers of the
        input bins.

    """
    hist = np.histogram(data, bins=bins, density=True)[0]
    kernel = KramersMoyalKernel(
        name=kernel_name,
        bandwidth=kernel_bandwidth,
        period=kernel_period,
    )
    hist_kde = get_kernel_density_estimate_from_histogram(hist, bins=[bins], kernel=kernel)
    return hist_kde


def process_dataframe_for_track_statistics(
    dataframe: pd.DataFrame, dataset_config: DatasetConfig, minimum_track_length: int | float
) -> pd.DataFrame:
    """
    Pre-process a DataFrame to prepare for track statistics calculations.

    This function performs the following steps:
        1. Filters the input DataFrame to include only timepoints annotated as
           "steady state" according to the provided dataset configuration.
        2. Adds a "track length" column to the DataFrame, which computes the
           length of each track (crop) as the difference between the maximum and
           minimum timepoints for that track. This ensures filtering by track
           length is done based on the track length within the steady state
           period, rather than the full track length.
        3. Filters the DataFrame to include only tracks that meet a minimum
           track length criterion.

    Parameters
    ----------
    dataframe
        The input DataFrame containing the time series data for multiple tracks
        (crops).
    dataset_config
        The dataset configuration object that contains information about the
        steady state timepoints.
    minimum_track_length
        The minimum track length (in time units) required for a track to be
        included in the final DataFrame. Tracks with a length shorter than this
        threshold will be excluded.

    Returns
    -------
    :
        A filtered DataFrame that includes only steady state timepoints and tracks
        that meet the minimum track length criterion within the steady state period.

    """
    # filter to steady state timepoints only
    dataframe_steady_state = filter_dataframe_to_steady_state(dataframe, dataset_config)

    # add track length column based on steady state timepoints only, then filter
    # by track length
    dataframe_with_duration = add_track_duration_to_dataframe(
        dataframe_steady_state, grouping_columns=[Column.CROP_INDEX], time_column=Column.TIMEPOINT
    )
    dataframe_min_length = filter_dataframe_by_track_length(
        dataframe_with_duration, minimum_track_length
    )
    return dataframe_min_length
