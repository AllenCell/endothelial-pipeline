import numpy as np
import pandas as pd

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    rewrap_polar_angle,
    unwrap_nonsequential_array,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName


def compute_circular_mean(
    angles: np.ndarray, original_angle_range: tuple[float, float], rewrap: bool = True
) -> float:
    """
    Compute the circular mean of a set of angles.

    Parameters
    ----------
    angles
        An array of angles from which to compute the circular mean.
    original_angle_range
        A tuple specifying the original range of the angles, e.g., (0, 360) for
        degrees or (0, 2*np.pi) for radians.
    rewrap
        If True, the resulting mean will be rewrapped to the original angle
        range. If False, the mean will be returned in the unwrapped form.
    """
    angle_period = original_angle_range[1] - original_angle_range[0]

    unwrapped_angles = unwrap_nonsequential_array(angles, angle_period)
    unwrapped_mean = np.mean(unwrapped_angles)

    if rewrap:
        return rewrap_polar_angle(unwrapped_mean, original_angle_range)
    else:
        return unwrapped_mean


def compute_circular_std(angles: np.ndarray, original_angle_range: tuple[float, float]) -> float:
    """
    Compute the circular standard deviation of a set of angles.

    Parameters
    ----------
    angles
        An array of angles from which to compute the circular standard deviation.
    original_angle_range
        A tuple specifying the original range of the angles, e.g., (0, 360) for
        degrees or (0, 2*np.pi) for radians.
    """
    angle_period = original_angle_range[1] - original_angle_range[0]

    unwrapped_angles = unwrap_nonsequential_array(angles, angle_period)
    unwrapped_std = np.std(unwrapped_angles)

    return unwrapped_std


def compute_circular_mean_and_std_over_time(
    df: pd.DataFrame,
    column_name: str,
    original_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute circular mean and standard deviation of a periodic column at each
    timepoint.

    Parameters
    ----------
    df
        Feature dataframe for a single dataset / flow condition, containing a
        ``frame_number`` column and the periodic feature column.
    column_name
        Name of the periodic feature column.
    original_range
        Original range of the periodic variable, passed to circmean and circstd
        to ensure correct handling of wraparound.

    Returns
    -------
    time_values
        1-D array of time values in hours.
    mean_values
        1-D array of rewrapped circular mean at each timepoint.
    std_values
        1-D array of standard deviation of the unwrapped values at each
        timepoint.
    """
    timepoints = df[ColumnName.TIMEPOINT.value].sort_values().unique()
    mean_values = np.empty(len(timepoints), dtype=float)
    std_values = np.empty(len(timepoints), dtype=float)

    for i, (_, df_frame) in enumerate(df.groupby(ColumnName.TIMEPOINT.value)):
        unwrapped_angles = df_frame[column_name].to_numpy()
        mean_values[i] = compute_circular_mean(unwrapped_angles, original_range, rewrap=True)
        std_values[i] = compute_circular_std(unwrapped_angles, original_range)

    return mean_values, std_values


def compute_per_crop_temporal_cov(
    crop_array: np.ndarray,
) -> np.ndarray:
    """
    Compute the temporal CoV (std / |mean| over time) for every individual crop.

    Missing timepoints are treated as NaN so that crops with gaps are still
    included.  Crops where |mean| close to 0 produce infinite or NaN CoV and are
    silently dropped from the returned arrays.

    Parameters
    ----------
    crop_array
        3-D array of shape (n_crops, n_timepoints, 1) containing the
        feature values for each crop and timepoint.
    """
    temporal_std = np.nanstd(crop_array, axis=1)
    temporal_mean_abs = np.abs(np.nanmean(crop_array, axis=1))
    cov = np.where(temporal_mean_abs > 0, temporal_std / temporal_mean_abs, np.nan)
    cov_finite = cov[np.isfinite(cov)]
    return cov_finite


def compute_cumulative_variance_ratio_vs_time(
    crop_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ratio of individual to population variance as a function of time.

    At each timepoint *T* the function computes:

    * **Population variance** — variance of the feature across all crops at *T*.
    * **Individual cumulative variance** — for each crop, the variance of the
      feature from timepoint 0 up to and including *T*.
    * **Ratio** — mean individual cumulative variance / population variance.

    A ratio close to 1 at all times indicates ergodic behaviour; deviations
    reveal non-ergodicity.

    Parameters
    ----------
    crop_array
        3-D array of shape (n_crops, n_timepoints, 1) containing the
        feature values for each crop and timepoint.  Missing timepoints should
        be represented as NaN so that crops with gaps are still included in the
        variance calculations.

    Returns
    -------
    ratio_mean
        1-D array of mean ratio of individual to population variance at each
        timepoint.
    ratio_upper
        1-D array of upper bound of mean ± SEM for the ratio at each timepoint.
    ratio_lower
        1-D array of lower bound of mean ± SEM for the ratio at each timepoint.
    """
    # shape: (n_crops, n_timepoints, 1)
    n_crops, n_timepoints, _ = crop_array.shape

    # population variance at each timepoint (across crops)
    pop_var = np.nanvar(crop_array, axis=0)  # (n_timepoints,)

    # per-crop cumulative temporal variance up to each timepoint
    # ind_cum_var[i, t] = Var(crop_feat[i, 0:t+1])
    ind_cum_var = np.full((n_crops, n_timepoints), np.nan)
    for t in range(n_timepoints):
        if t == 0:
            # variance of a single value is 0
            ind_cum_var[:, t] = 0.0
        else:
            ind_cum_var[:, t] = np.nanvar(crop_array[:, : t + 1], axis=1).flatten()

    # mean and SEM of individual cumulative variance across crops
    mean_ind_var = np.nanmean(ind_cum_var, axis=0)  # (n_timepoints,)
    n_valid = np.sum(np.isfinite(ind_cum_var), axis=0)
    sem_ind_var = np.where(
        n_valid > 1,
        np.nanstd(ind_cum_var, axis=0) / np.sqrt(n_valid),
        0.0,
    )

    # ratio = mean individual var / population var
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_mean = np.where(pop_var > 0, mean_ind_var / pop_var, np.nan)
        ratio_upper = np.where(pop_var > 0, (mean_ind_var + sem_ind_var) / pop_var, np.nan)
        ratio_lower = np.where(pop_var > 0, (mean_ind_var - sem_ind_var) / pop_var, np.nan)

    return ratio_mean, ratio_upper, ratio_lower


def compute_binned_variance_ratio_vs_time(
    crop_array: np.ndarray,
    bin_size: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ratio of individual to population variance within non-overlapping
    time bins.

    This is the *binned* (non-cumulative) counterpart of
    :func:`compute_variance_ratio_vs_time`.  Instead of accumulating variance
    from the start up to each timepoint, this function partitions the time axis
    into fixed-width bins and computes variances **within each bin only**.

    At each time bin the function computes:

    * **Population variance** — variance of the feature across all crops and all
      timepoints that fall in the bin.
    * **Individual bin variance** — for each crop, the variance of the feature
      over the timepoints within the bin.
    * **Ratio** — mean individual bin variance / population variance.

    **Interpretation**

    * A ratio near **1** at all bin centres indicates that a single crop's
      temporal variability within a short window is comparable to the spread
      across the population — consistent with **ergodic** behaviour.
    * A ratio **<< 1** means individual crops explore only a small fraction of
      the population-level spread within each window, indicating **non-ergodic**
      or **heterogeneous** dynamics where different crops occupy distinct
      regions of feature space.
    * A ratio that **increases toward 1** over time suggests that the system is
      *mixing* — crops start in distinct states but progressively explore more
      of the available feature space.
    * Unlike the cumulative variant, the binned ratio is sensitive to
      *local-in-time* fluctuations and is not biased by early transient
      dynamics.  Comparing cumulative and binned ratios therefore reveals
      whether ergodicity is driven by long-term drift or local fluctuations.

    Parameters
    ----------
    crop_array
        3-D array of shape (n_crops, n_timepoints, 1) containing the feature
        values for each crop and timepoint.  Missing timepoints should be
        represented as NaN so that crops with gaps are still included in the
        variance calculations.
    bin_size
        Width of binning window in time steps (frames).

    Returns
    -------
    bin_centres
        1-D array of time values corresponding to the centre of each bin.
    ratio_mean
        1-D array of mean ratio of individual to population variance for each
        bin.
    ratio_upper
        1-D array of upper bound of mean ± SEM for the ratio in each bin.
    ratio_lower
        1-D array of lower bound of mean ± SEM for the ratio in each bin.
    """
    # shape: (n_crops, n_timepoints, n_features)
    n_timepoints = crop_array.shape[1]

    # convert bin size from hours to frames
    bin_edges = np.arange(0, n_timepoints + 1, bin_size)
    # ensure last bin edge covers remaining frames
    if bin_edges[-1] < n_timepoints:
        bin_edges = np.append(bin_edges, n_timepoints)
    n_bins = len(bin_edges) - 1
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    ratio_mean = np.full(n_bins, np.nan)
    ratio_upper = np.full(n_bins, np.nan)
    ratio_lower = np.full(n_bins, np.nan)

    for b in range(n_bins):
        t_start = bin_edges[b]
        t_end = bin_edges[b + 1]
        bin_data = crop_array[:, t_start:t_end]  # (n_crops, bin_width)

        if bin_data.shape[1] < 2:
            # cannot compute variance from a single timepoint
            continue

        # population variance: flatten all crops x timepoints in this bin
        pop_var = np.nanvar(bin_data)

        # per-crop variance within this bin
        ind_var = np.nanvar(bin_data, axis=1)  # (n_crops,)

        mean_ind_var = np.nanmean(ind_var)
        n_valid = np.sum(np.isfinite(ind_var))
        sem_ind_var = np.nanstd(ind_var) / np.sqrt(n_valid) if n_valid > 1 else 0.0

        if pop_var > 0:
            ratio_mean[b] = mean_ind_var / pop_var
            ratio_upper[b] = (mean_ind_var + sem_ind_var) / pop_var
            ratio_lower[b] = (mean_ind_var - sem_ind_var) / pop_var

    return bin_centres, ratio_mean, ratio_upper, ratio_lower
