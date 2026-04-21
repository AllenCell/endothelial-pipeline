"""Methods related to computing long-time-scale statistics of time series data."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.interpolate import make_interp_spline

from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    get_kernel_density_estimate_from_histogram,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins


def compute_kde_on_bins(
    data: np.ndarray,
    bin_width: float,
    kernel_name: Literal["gaussian", "epanechnikov", "periodic"],
    kernel_bandwidth: float,
    kernel_period: float | None,
    bins: np.ndarray | None = None,
    pad_bins: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a kernel density estimate (KDE) on the native histogram bin centers.

    Unlike :func:`compute_interpolated_kde_spline`, this function returns KDE
    values on the coarse bin-center grid without any spline smoothing.  This
    is the right representation for accumulating bootstrap samples: average the
    raw KDE values first, then apply spline smoothing only once at plot time
    via :func:`smooth_kde_with_spline`.

    Parameters
    ----------
    data
        1D array of data points to estimate the density for.
    bin_width
        The width of the histogram bins used to compute the KDE.
    kernel_name
        The name of the kernel to use for the KDE.
    kernel_bandwidth
        The bandwidth parameter for the kernel density estimate.
    kernel_period
        The period for periodic kernels (pass None for non-periodic kernels).
    bins
        Pre-computed bin edges (1D array) to use instead of deriving them from
        ``data``.  Pass this to ensure all bootstrap samples share the same
        bin grid so their KDE arrays can be stacked and averaged.  When
        ``None``, bin edges are derived from ``data``.
    pad_bins
        Amount to pad histogram bins on either side of the data range.
        Ignored when ``bins`` is provided.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bin_centers, kde_values)`` — both 1D arrays of the same length.

    """
    if bins is not None:
        computed_bins = [bins]
        centers = [(bins[:-1] + bins[1:]) / 2]
    else:
        computed_bins, centers = get_bins(bin_widths=(bin_width,), data=data, pad=pad_bins)
    hist = np.histogram(data, bins=computed_bins[0], density=True)[0]
    kernel = KramersMoyalKernel(
        name=kernel_name,
        bandwidth=kernel_bandwidth,
        period=kernel_period,
    )
    hist_kde = get_kernel_density_estimate_from_histogram(hist, bins=computed_bins, kernel=kernel)
    return centers[0], hist_kde


def smooth_kde_with_spline(
    bin_centers: np.ndarray,
    kde_values: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """Fit a cubic spline to a KDE and evaluate it on a fine grid.

    Intended to be called at *plot time* on KDE values that were first computed
    (and optionally averaged / CI-bounded) on a coarse bin-center grid via
    :func:`compute_kde_on_bins`.

    Parameters
    ----------
    bin_centers
        1D array of bin-center x-values (the coarse grid).
    kde_values
        1D array of KDE values at each bin center.
    x_eval
        1D array of x-values at which to evaluate the smoothed spline.

    Returns
    -------
    np.ndarray
        KDE values evaluated at each point in ``x_eval``.  Positions where
        ``kde_values`` is not finite are excluded from the spline fit.
        Returns all-NaN if fewer than four finite values exist.

    """
    finite_mask = np.isfinite(kde_values)
    if finite_mask.sum() < 4:
        return np.full(len(x_eval), np.nan)
    knot_x = bin_centers[finite_mask]
    spline = make_interp_spline(knot_x, kde_values[finite_mask], k=3)
    return spline(x_eval)


def compute_cumulative_variance_over_time(
    crop_array: np.ndarray, variance_function: Callable[..., float], **var_func_kwargs
) -> np.ndarray:
    """Compute per-crop cumulative variance of a feature over time.

    **Handling of NaN values**

    If the input ``crop_array`` contains NaN values (e.g., due to missing
    timepoints for some crops), the function will handle them as follows:
        - If all values for a crop up to a given timepoint are NaN, the
          cumulative variance for that crop at that timepoint will be set to NaN
        - If some (but not all) values for a crop up to a given timepoint are
          NaN, the variance function will be applied to all values.

    Thus, if the variance function can handle NaN values (e.g., by ignoring
    them), then the cumulative variance will be computed using the available
    data. If the variance function does not handle NaN values, it may return NaN
    for that crop and timepoint.

    Parameters
    ----------
    crop_array
        2-D array containing the feature values for each crop and timepoint.
    variance_function
        Function to compute the variance. Should accept a 1-D array and return a
        float.
    **var_func_kwargs
        Additional keyword arguments to pass to the variance function.

    Returns
    -------
    :
        2-D array of the same shape as ``crop_array``, where each element [i, t] contains
        the variance of the feature for crop i computed from time 0 up to time t.

    """
    cumulative_var_per_crop = np.zeros_like(crop_array)  # shape: (n_crops, n_timepoints)
    for i in range(crop_array.shape[1]):
        if i == 0:
            # cannot compute variance from a single timepoint, so set to 0 and
            # skip to next iteration
            continue
        else:
            data_to_t = crop_array[:, : i + 1]  # all crops from time 0 to i
            if np.isnan(data_to_t).all():
                # if all values are NaN, skip variance calculation and set to NaN
                cumulative_var_per_crop[:, i] = np.nan
                continue
            # for each crop, compute variance of the feature from time 0 to i;
            # only apply to rows that are not entirely NaN to avoid RuntimeWarnings
            result = np.full(data_to_t.shape[0], np.nan)
            valid_rows = ~np.isnan(data_to_t).all(axis=1)
            if valid_rows.any():
                result[valid_rows] = variance_function(
                    data_to_t[valid_rows], **var_func_kwargs, axis=1
                )
            cumulative_var_per_crop[:, i] = result

    # where data are missing, set to NaN
    cumulative_var_per_crop[~np.isfinite(crop_array)] = np.nan
    return cumulative_var_per_crop


def compute_binned_variance_ratio_vs_time(
    crop_array: np.ndarray,
    bin_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ratio of individual to population variance within non-overlapping time bins.

    This is the *binned* (non-cumulative) counterpart of
    :func:`compute_variance_ratio_vs_time`.  Instead of accumulating variance
    from the start up to each timepoint, this function partitions the time axis
    into fixed-width bins and computes variances **within each bin only**.

    At each time bin the function computes:

        - **Population variance:** variance of the feature across all crops and
          all timepoints that fall in the bin.
        - **Individual bin variance:** for each crop, the variance of the
          feature over the timepoints within the bin.
        - **Ratio:** mean individual bin variance / population variance.

    **Interpretation**

    Interpreting the binned variance ratio over time provides insights into the
    temporal dynamics and ergodicity of the system:

        - A ratio near **1** at all bin centres indicates that a single crop's
          temporal variability within a short window is comparable to the spread
          across the population — consistent with **ergodic** behaviour.
        - A ratio **<< 1** means individual crops explore only a small fraction
          of the population-level spread within each window, indicating
          **non-ergodic** or **heterogeneous** dynamics where different crops
          occupy distinct regions of feature space.
        - A ratio that **increases toward 1** over time suggests that the system
          is *mixing* — crops start in distinct states but progressively explore
          more of the available feature space.
        - Unlike the cumulative variant, the binned ratio is sensitive to
          *local-in-time* fluctuations and is not biased by early transient
          dynamics.

    Comparing cumulative and binned ratios therefore reveals whether ergodicity
    is driven by long-term drift or local fluctuations.

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
    :
        1-D array of time values corresponding to the centre of each bin.
    :
        1-D array of mean ratio of individual to population variance for each
        bin.
    :
        1-D array of upper bound of mean ± SEM for the ratio in each bin.
    :
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

        # per-crop variance within this bin; mask all-NaN crop rows first
        valid_crop_rows = ~np.isnan(bin_data).all(axis=1)
        ind_var = np.full(bin_data.shape[0], np.nan)
        if valid_crop_rows.any():
            ind_var[valid_crop_rows] = np.nanvar(bin_data[valid_crop_rows], axis=1)  # (n_crops,)

        n_valid = np.sum(np.isfinite(ind_var))
        if n_valid == 0:
            continue
        mean_ind_var = np.nanmean(ind_var)
        sem_ind_var = np.nanstd(ind_var) / np.sqrt(n_valid) if n_valid > 1 else 0.0

        if pop_var > 0:
            ratio_mean[b] = mean_ind_var / pop_var
            ratio_upper[b] = (mean_ind_var + sem_ind_var) / pop_var
            ratio_lower[b] = (mean_ind_var - sem_ind_var) / pop_var

    return bin_centres, ratio_mean, ratio_upper, ratio_lower
