import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.io import get_output_path, load_image, save_plot_to_path
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.method_constants import GFP_ROLLING_WINDOW, OUTLIER_THRESHOLD


def plot_gfp_outliers_rolling(
    tp_means: np.ndarray,
    rolling_mean: np.ndarray,
    lower_threshold: np.ndarray,
    upper_threshold: np.ndarray,
    dark_outliers: list[int],
    bright_outliers: list[int],
    dataset_name: str,
    position: int,
    percent: float = OUTLIER_THRESHOLD,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH / 2, 3),
) -> None:
    """
    Plot timepoint-level mean intensities with rolling mean ± percentage thresholds and outliers.

    Parameters
    ----------
    tp_means : numpy.ndarray
        Array of mean intensities for each timepoint.
    rolling_mean : numpy.ndarray
        Array of rolling mean values for the timepoints.
    lower_threshold : numpy.ndarray
        Array of lower threshold values for the timepoints.
    upper_threshold : numpy.ndarray
        Array of upper threshold values for the timepoints.
    dark_outliers : list of int
        Indices of timepoints identified as dark outliers.
    bright_outliers : list of int
        Indices of timepoints identified as bright outliers.
    dataset_name : str
        Name of the dataset being analyzed.
    position : int
        Position index within the dataset.
    window : int, optional
        Size of the rolling window used for calculating the rolling mean (default is GFP_ROLLING_WINDOW).
    percent : float, optional
        Threshold percentage for identifying outliers (default is THRESHOLD).
    """

    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(tp_means, label="Intensity", color="black", alpha=0.7)
    ax.plot(rolling_mean, label="Rolling mean", color="blue", alpha=0.9)
    ax.plot(lower_threshold, color="red", linestyle="--", label=f"Lower {int(percent*100)}%")
    ax.plot(upper_threshold, color="orange", linestyle="--", label=f"Upper {int(percent*100)}%")

    if dark_outliers:
        ax.scatter(
            dark_outliers, tp_means[dark_outliers], color="red", label="Dark outliers", zorder=5
        )

    if bright_outliers:
        ax.scatter(
            bright_outliers,
            tp_means[bright_outliers],
            color="orange",
            label="Bright outliers",
            zorder=5,
        )

    info_lines = []
    if dark_outliers:
        info_lines.append(f"Dark outliers: {dark_outliers}")
    if bright_outliers:
        info_lines.append(f"Bright outliers: {bright_outliers}")

    if info_lines:
        print("\n".join(info_lines))

    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Mean VE-cadherin\nintensity in Z-stack (a.u.)")
    ax.tick_params(axis="both", which="major")

    ncols = 4 if not dark_outliers and not bright_outliers else 3
    (lines, labels) = plt.gca().get_legend_handles_labels()
    ax.legend(
        lines,
        labels,
        loc="upper left",
        ncol=ncols,
        fontsize=FONTSIZE_XSMALL,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
        borderpad=0.25,
        borderaxespad=0.25,
    )

    # reduce label padding
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3

    save_dir = get_output_path("annotate_tp_outliers")
    save_plot_to_path(fig, save_dir, f"gfp_outliers_{dataset_name}_P{position}", file_format=".svg")


def detect_egfp_scope_errors(
    dataset_config: DatasetConfig,
    position: int,
    visualize: bool = False,
    window: int = GFP_ROLLING_WINDOW,
    percent: float = OUTLIER_THRESHOLD,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH / 2, 3),
) -> dict[ColumnNameType, int | list[int] | list[float] | np.ndarray]:
    """
    Detect EGFP scope errors based on per-timepoint mean with rolling mean ±
    percentage thresholds.

    This function computes the mean intensity for each timepoint using all
    z-slices and identifies outlier timepoints based on a rolling mean and
    percentage thresholds. Optionally, it can visualize the results.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Configuration object containing dataset information and parameters.
    position : int
        Position index within the dataset to analyze.
    visualize : bool, optional
        If True, generates a visualization of the detected outliers (default is False).
    window : int, optional
        Size of the rolling window used for calculating the rolling mean (default is GFP_ROLLING_WINDOW).
    percent : float, optional
        Threshold percentage for identifying outliers (default is OUTLIER_THRESHOLD).
    figure_size
        The size of the figure to generate if visualize is True (default is (MAX_FIGURE_WIDTH/2, 3)).

    Returns
    -------
    list of int
        Indices of timepoints identified as outliers.
    """

    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    gfp_zarr = load_image(zarr_loc, channels=["EGFP"], level=1, squeeze=True)

    # Compute mean intensity across spatial dimensions (Y, X)
    intensity_array = gfp_zarr.mean(axis=(-2, -1))  # now (T, Z)

    # Compute per-timepoint mean (across Z)
    tp_means = intensity_array.mean(axis=1).compute().astype(float)  # shape (T,)

    # Rolling median
    series = pd.Series(tp_means)
    rolling_median = series.rolling(window, center=True).median()

    # Pad edges
    start_val = np.nanmedian(tp_means[:window])
    end_val = np.nanmedian(tp_means[-window:])
    rolling_median.iloc[: window // 2] = start_val
    rolling_median.iloc[-window // 2 :] = end_val
    rolling_median = rolling_median.to_numpy()

    # Thresholds
    lower_threshold = rolling_median * (1 - percent)
    upper_threshold = rolling_median * (1 + percent)

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
