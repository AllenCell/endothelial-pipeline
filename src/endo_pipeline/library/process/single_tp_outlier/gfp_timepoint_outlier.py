import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.io import get_output_path, load_image, save_plot_to_path
from endo_pipeline.manifests import get_zarr_location_for_position
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
    window: int = GFP_ROLLING_WINDOW,
    percent: float = OUTLIER_THRESHOLD,
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

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(tp_means, label="TP Mean Intensity", color="black", alpha=0.7)
    ax.plot(rolling_mean, label=f"Rolling Mean (window={window})", color="blue", alpha=0.9)
    ax.plot(lower_threshold, color="red", linestyle="--", label=f"Lower ({int(percent*100)}%)")
    ax.plot(upper_threshold, color="orange", linestyle="--", label=f"Upper ({int(percent*100)}%)")

    if dark_outliers:
        ax.scatter(
            dark_outliers, tp_means[dark_outliers], color="red", label="Dark Outliers", zorder=5
        )
    if bright_outliers:
        ax.scatter(
            bright_outliers,
            tp_means[bright_outliers],
            color="orange",
            label="Bright Outliers",
            zorder=5,
        )

    info_lines = []
    if dark_outliers:
        info_lines.append(f"Dark: {dark_outliers}")
    if bright_outliers:
        info_lines.append(f"Bright: {bright_outliers}")

    if info_lines:
        fig.text(
            1.02,
            0.5,
            "\n".join(info_lines),
            fontsize=12,
            va="center",
            ha="left",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Time (frames)", fontsize=14, labelpad=10)
    ax.set_ylabel("Average mEGFP intensity in Z-stack (a.u.)", fontsize=14, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12, loc="upper right", frameon=True)

    fig.tight_layout(rect=[0, 0, 0.8, 1])

    save_dir = get_output_path("annotate_tp_outliers")
    save_plot_to_path(fig, save_dir, f"gfp_outliers_{dataset_name}_P{position}", file_format=".pdf")
    plt.show()
    plt.close(fig)


def detect_egfp_scope_errors(
    dataset_config: DatasetConfig,
    position: int,
    visualize: bool = False,
    window: int = GFP_ROLLING_WINDOW,
    percent: float = OUTLIER_THRESHOLD,
) -> list[int]:
    """
    Detect EGFP scope errors based on per-timepoint mean with rolling mean ± percentage thresholds.

    This function computes the mean intensity for each timepoint using all z-slices and identifies
    outlier timepoints based on a rolling mean and percentage thresholds. Optionally, it can
    visualize the results.

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

    if visualize:
        plot_gfp_outliers_rolling(
            tp_means,
            rolling_median,
            lower_threshold,
            upper_threshold,
            dark_outliers,
            bright_outliers,
            dataset_config.name,
            position,
            window=window,
            percent=percent,
        )

    return egfp_scope_error
