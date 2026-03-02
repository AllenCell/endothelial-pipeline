import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd

from endo_pipeline.library.analyze.diffae_dataframe_utils import df_to_array
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName


def compute_circular_mean_std(
    df: pd.DataFrame,
    column_name: str,
    original_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    time_step_minutes
        Duration of one frame in minutes, used to convert frame indices to
        hours.
    period
        Period of the periodic variable (e.g. ``pi`` for rescaled theta).
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
        mean_values[i] = circmean(unwrapped_angles, low=original_range[0], high=original_range[1])
        std_values[i] = circstd(unwrapped_angles, low=original_range[0], high=original_range[1])

    return mean_values, std_values


def compute_population_mean_std(
    df: pd.DataFrame,
    column_names: list[str],
    time_step_minutes: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Compute population mean and standard deviation at each timepoint.

    Parameters
    ----------
    df
        Feature dataframe for a single dataset / flow condition, containing a
        ``frame_number`` column and the feature columns listed in
        ``column_names``.
    column_names
        Feature column names to compute statistics for.
    time_step_minutes
        Duration of one frame in minutes, used to convert frame indices to
        hours.

    Returns
    -------
    time_values
        1-D array of time values in hours.
    mean_df
        DataFrame of mean values indexed by frame number, with one column per
        feature.
    std_df
        DataFrame of standard-deviation values indexed by frame number, with
        one column per feature.
    """
    grouped = df.groupby(ColumnName.TIMEPOINT.value)[column_names]
    mean_df: pd.DataFrame = grouped.mean()
    std_df: pd.DataFrame = grouped.std()
    time_values: np.ndarray = mean_df.index.values * time_step_minutes / 60
    return time_values, mean_df, std_df


def compute_population_cov(
    df: pd.DataFrame,
    column_names: list[str],
    time_step_minutes: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute the population (ensemble) coefficient of variation at each timepoint.

    **CAUTION**: This function is not designed to handle features whose mean across crops
    is close to zero, as this produces artificially high CoV values. Interpret results
    with caution.

    The population CoV is defined as std / |mean| across all crops at each timepoint.
    Using the absolute value of the mean prevents sign-flip artefacts for features
    whose ensemble mean passes through zero.

    Parameters
    ----------
    df
        Feature dataframe for a single dataset / flow condition, containing a
        ``frame_number`` column and the feature columns listed in ``column_names``.
    column_names
        Feature column names to compute CoV for.
    time_step_minutes
        Duration of one frame in minutes, used to convert frame indices to hours.

    Returns
    -------
    time_values
        1-D array of time values in hours.
    cov_df
        DataFrame of CoV values indexed by frame number, with one column per feature.
    """
    grouped = df.groupby(ColumnName.TIMEPOINT.value)[column_names]
    cov_df: pd.DataFrame = grouped.std() / grouped.mean().abs()
    time_values: np.ndarray = cov_df.index.values * time_step_minutes / 60
    return time_values, cov_df


def compute_per_crop_temporal_cov(
    df: pd.DataFrame,
    column_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Compute the temporal CoV (std / |mean| over time) for every individual crop.

    Missing timepoints are treated as NaN so that crops with gaps are still
    included.  Crops where |mean| close to 0 produce infinite or NaN CoV and are
    silently dropped from the returned arrays.

    Parameters
    ----------
    df
        Feature dataframe for a single dataset / flow condition.  Must contain
        ``crop_index`` and ``frame_number`` columns.
    column_names
        Feature column names to compute CoV for.

    Returns
    -------
    :
        Mapping from feature column name to 1-D array of finite per-crop temporal
        CoV values.
    """
    # shape: (n_crops, n_timepoints, n_features)
    crop_array = df_to_array(df, column_names)

    crop_cov_dict: dict[str, np.ndarray] = {}
    for feat_idx, col in enumerate(column_names):
        crop_feat = crop_array[..., feat_idx]  # (n_crops, n_timepoints)
        temporal_std = np.nanstd(crop_feat, axis=1)
        temporal_mean_abs = np.abs(np.nanmean(crop_feat, axis=1))
        cov = np.where(temporal_mean_abs > 0, temporal_std / temporal_mean_abs, np.nan)
        crop_cov_dict[col] = cov[np.isfinite(cov)]

    return crop_cov_dict


def compute_cumulative_variance_ratio_vs_time(
    df: pd.DataFrame,
    column_names: list[str],
    time_step_minutes: float,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
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
    df
        Feature dataframe for a single dataset / flow condition.  Must contain
        ``crop_index`` and ``frame_number`` columns.
    column_names
        Feature column names to compute the variance ratio for.
    time_step_minutes
        Duration of one frame in minutes, used to convert frame indices to
        hours.

    Returns
    -------
    time_values
        1-D array of time values in hours (one per timepoint).
    ratio_dict
        Mapping from feature column name to a tuple
        ``(ratio_mean, ratio_upper, ratio_lower)`` where each element is a
        1-D array aligned with *time_values*.  ``ratio_upper`` and
        ``ratio_lower`` are the mean ± SEM bounds.
    """
    # shape: (n_crops, n_timepoints, n_features)
    crop_array = df_to_array(df, column_names)
    n_crops, n_timepoints, _ = crop_array.shape

    timepoints = np.arange(n_timepoints)
    time_values = timepoints * time_step_minutes / 60

    ratio_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for feat_idx, col in enumerate(column_names):
        crop_feat = crop_array[..., feat_idx]  # (n_crops, n_timepoints)

        # population variance at each timepoint (across crops)
        pop_var = np.nanvar(crop_feat, axis=0)  # (n_timepoints,)

        # per-crop cumulative temporal variance up to each timepoint
        # ind_cum_var[i, t] = Var(crop_feat[i, 0:t+1])
        ind_cum_var = np.full((n_crops, n_timepoints), np.nan)
        for t in range(n_timepoints):
            if t == 0:
                # variance of a single value is 0
                ind_cum_var[:, t] = 0.0
            else:
                ind_cum_var[:, t] = np.nanvar(crop_feat[:, : t + 1], axis=1)

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

        ratio_dict[col] = (ratio_mean, ratio_upper, ratio_lower)

    return time_values, ratio_dict


def compute_binned_variance_ratio_vs_time(
    df: pd.DataFrame,
    column_names: list[str],
    time_step_minutes: float,
    bin_size_hours: int = 2,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Compute ratio of individual to population variance within non-overlapping time bins.

    This is the *binned* (non-cumulative) counterpart of
    :func:`compute_variance_ratio_vs_time`.  Instead of accumulating variance
    from the start up to each timepoint, this function partitions the time axis
    into fixed-width bins and computes variances **within each bin only**.

    At each time bin the function computes:

    * **Population variance** — variance of the feature across all crops and
      all timepoints that fall in the bin.
    * **Individual bin variance** — for each crop, the variance of the feature
      over the timepoints within the bin.
    * **Ratio** — mean individual bin variance / population variance.

    **Interpretation**

    * A ratio near **1** at all bin centres indicates that a single crop's
      temporal variability within a short window is comparable to the spread
      across the population — consistent with **ergodic** behaviour.
    * A ratio **<< 1** means individual crops explore only a small fraction of
      the population-level spread within each window, indicating
      **non-ergodic** or **heterogeneous** dynamics where different crops
      occupy distinct regions of feature space.
    * A ratio that **increases toward 1** over time suggests that the system is
      *mixing* — crops start in distinct states but progressively explore more
      of the available feature space.
    * Unlike the cumulative variant, the binned ratio is sensitive to
      *local-in-time* fluctuations and is not biased by early transient
      dynamics.  Comparing cumulative and binned ratios therefore reveals
      whether ergodicity is driven by long-term drift or local fluctuations.

    Parameters
    ----------
    df
        Feature dataframe for a single dataset / flow condition.  Must contain
        ``crop_index`` and ``frame_number`` columns.
    column_names
        Feature column names to compute the variance ratio for.
    time_step_minutes
        Duration of one frame in minutes, used to convert bin centres to hours.
    bin_size_hours
        Width of each non-overlapping time bin in hours.

    Returns
    -------
    bin_centres_hours
        1-D array of bin-centre time values in hours.
    ratio_dict
        Mapping from feature column name to a tuple
        ``(ratio_mean, ratio_upper, ratio_lower)`` where each element is a
        1-D array aligned with *bin_centres_hours*.  ``ratio_upper`` and
        ``ratio_lower`` are the mean ± SEM bounds.
    """
    # shape: (n_crops, n_timepoints, n_features)
    crop_array = df_to_array(df, column_names)
    n_crops, n_timepoints, _ = crop_array.shape

    # convert bin size from hours to frames
    bin_size_frames = int(bin_size_hours * 60 / time_step_minutes)
    bin_edges = np.arange(0, n_timepoints + 1, bin_size_frames)
    # ensure last bin edge covers remaining frames
    if bin_edges[-1] < n_timepoints:
        bin_edges = np.append(bin_edges, n_timepoints)
    n_bins = len(bin_edges) - 1
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_centres_hours = bin_centres * time_step_minutes / 60

    ratio_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for feat_idx, col in enumerate(column_names):
        crop_feat = crop_array[..., feat_idx]  # (n_crops, n_timepoints)

        ratio_mean = np.full(n_bins, np.nan)
        ratio_upper = np.full(n_bins, np.nan)
        ratio_lower = np.full(n_bins, np.nan)

        for b in range(n_bins):
            t_start = bin_edges[b]
            t_end = bin_edges[b + 1]
            bin_data = crop_feat[:, t_start:t_end]  # (n_crops, bin_width)

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

        ratio_dict[col] = (ratio_mean, ratio_upper, ratio_lower)

    return bin_centres_hours, ratio_dict
