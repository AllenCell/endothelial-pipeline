"""Methods for detecting single timepoint outliers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from endo_pipeline.configs import (
    ChannelName,
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    load_dataset_config,
)
from endo_pipeline.io import load_image, save_plot_to_path
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
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
    bf_zarr = load_image(zarr_loc, channels=[ChannelName.BF], level=1, squeeze=True)

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
    gfp_zarr = load_image(zarr_loc, channels=[ChannelName.EGFP], level=1, squeeze=True)

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


def plot_single_timepoint_bf_outliers(
    mean_intensity: np.ndarray,
    rolling_median: np.ndarray,
    dark_threshold: np.ndarray,
    bright_threshold: np.ndarray,
    dark_outliers: list[int],
    bright_outliers: list[int],
    dataset_name: str,
    position: int,
    save_dir: Path,
    num_zslices: int = NUM_ZSLICES,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH / 2, 3),
) -> None:
    """
    Plot intensity data with thresholds and outliers, embedding outlier info on
    the plot.

    Parameters
    ----------
    mean_intensity
        The intensity data flattened indices of timepoints and z-slices.
    rolling_median
        The rolling median of the intensity data.
    dark_threshold
        The lower threshold for detecting dark outliers.
    bright_threshold
        The upper threshold for detecting bright outliers.
    dark_outliers
        Indices of dark outliers in the data (includes partial dark outliers).
    bright_outliers
        Indices of bright outliers in the data.
    dataset_name
        The name of the dataset being analyzed, used for labeling the plot.
    position
        The position identifier within the dataset, used for labeling the plot.
    save_dir
        The directory to save the plot to.
    num_zslices
        The number of z-slices per timepoint.
    figure_size
        The size of the figure to generate.
    """

    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")

    ax.plot(mean_intensity, label="Intensity", color="black", alpha=0.5)
    ax.plot(
        rolling_median,
        label="Rolling median",
        color="black",
        alpha=1,
        zorder=4,
    )
    ax.plot(
        dark_threshold,
        label=f"Lower {PARTIAL_DARK_THRESHOLD*100}%",
        color="red",
        linestyle="--",
    )
    ax.plot(
        bright_threshold,
        label=f"Upper {OUTLIER_THRESHOLD*100}%",
        color="orange",
        linestyle="--",
    )

    ax.scatter(
        dark_outliers,
        mean_intensity[dark_outliers],
        color="red",
        label="Dark BF outliers",
        zorder=5,
    )
    ax.scatter(
        bright_outliers,
        mean_intensity[bright_outliers],
        color="orange",
        label="Bright BF outliers",
        zorder=5,
    )

    outlier_groups = [
        ("Dark", dark_outliers),
        ("Bright", bright_outliers),
    ]

    info_lines = ["timepoint: [z-slices]\n"]
    for title, indices in outlier_groups:
        d = {
            t: [i % num_zslices for i in indices if i // num_zslices == t]
            for t in sorted({i // num_zslices for i in indices})
        }
        if d:
            info_lines.append(f"{title}:\n" + "\n".join(f"{t}: {z}" for t, z in d.items()))

    if len(info_lines) > 1:
        print("\n\n".join(info_lines))

    mean_for_lim = np.mean(mean_intensity)
    ax.set_xlabel("Index (flattened Z-slices)")
    ax.set_ylabel("Mean BF\nintensity in Z-slice (a.u.)")

    ax.tick_params(axis="both", which="major")
    ax.tick_params(axis="both", which="minor")

    # Secondary X-axis for timepoints
    def index_to_tp(x):
        return x // num_zslices

    def tp_to_index(t):
        return t * num_zslices

    secax = ax.secondary_xaxis("top", functions=(index_to_tp, tp_to_index))
    secax.set_xlabel("Time (frames)")
    max_tp = mean_intensity.shape[0] // num_zslices
    secax.set_xticks(np.arange(0, max_tp + 1, 50))
    secax.tick_params(axis="x")

    ax.legend(
        loc="upper center",
        ncol=3,
        fontsize=FONTSIZE_XSMALL,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    ax.set_ylim(mean_for_lim - mean_for_lim * 0.05, mean_for_lim + mean_for_lim * 0.035)

    # reduce label padding
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3
    secax.xaxis.labelpad = 3

    save_plot_to_path(
        fig,
        save_dir,
        f"bf_outliers_{dataset_name}_P{position}",
        file_format=".svg",
        tight_layout=False,
        transparent=True,
    )


def plot_single_timepoint_gfp_outliers(
    timepoint_means: np.ndarray,
    rolling_median: np.ndarray,
    lower_threshold: np.ndarray,
    upper_threshold: np.ndarray,
    dark_outliers: list[int],
    bright_outliers: list[int],
    dataset_name: str,
    position: int,
    save_dir: Path,
    percent: float = OUTLIER_THRESHOLD,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH / 2, 3),
) -> None:
    """
    Plot timepoint-level mean intensities with rolling mean ± percentage
    thresholds and outliers.

    Parameters
    ----------
    timepoint_means
        Array of mean intensities for each timepoint.
    rolling_median
        Array of rolling median values for the timepoints.
    lower_threshold
        Array of lower threshold values for the timepoints.
    upper_threshold
        Array of upper threshold values for the timepoints.
    dark_outliers
        Indices of timepoints identified as dark outliers.
    bright_outliers
        Indices of timepoints identified as bright outliers.
    dataset_name
        Name of the dataset being analyzed.
    position
        Position index within the dataset.
    save_dir
        The directory to save the plot to.
    percent
        Threshold percentage for identifying outliers.
    figure_size
        The size of the figure to generate.
    """

    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(timepoint_means, label="Intensity", color="black", alpha=0.7)
    ax.plot(rolling_median, label="Rolling median", color="blue", alpha=0.9)
    ax.plot(lower_threshold, color="red", linestyle="--", label=f"Lower {int(percent*100)}%")
    ax.plot(upper_threshold, color="orange", linestyle="--", label=f"Upper {int(percent*100)}%")

    if dark_outliers:
        ax.scatter(
            dark_outliers,
            timepoint_means[dark_outliers],
            color="red",
            label="Dark outliers",
            zorder=5,
        )

    if bright_outliers:
        ax.scatter(
            bright_outliers,
            timepoint_means[bright_outliers],
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
        columnspacing=0.5,
        borderpad=0.25,
        borderaxespad=0.25,
    )

    # reduce label padding
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3

    save_plot_to_path(
        fig,
        save_dir,
        f"gfp_outliers_{dataset_name}_P{position}",
        file_format=".svg",
        transparent=True,
    )


def print_timepoint_annotation_performance_stats(
    datasets: list[str],
    manual_annotations: list[TimepointAnnotation],
    auto_annotations: list[TimepointAnnotation],
    annotation_type: str,
) -> str:
    """
    Calculate and print statistics for manual and automatic timepoint annotations.

    This function processes a list of datasets, compares manual and automatic annotations
    for each dataset and position, and calculates statistics such as the number of missed
    timepoints, total annotated timepoints, and artifact percentages.

    Parameters
    ----------
    datasets : list of str
        A list of dataset names to process.
    manual_annotations : list of TimepointAnnotation
        A list of manual annotation types to consider.
    auto_annotations : list of TimepointAnnotation
        A list of automatic annotation types to consider.
    annotation_type : str
        A string indicating the type of annotation (e.g., "Brightfield" or "GFP")
        for labeling the output statistics.

    Returns
    -------
    results: str
        A formatted string containing the calculated statistics.
    """

    stats = []
    for dataset_name in datasets:
        dataset_config = load_dataset_config(dataset_name)
        for position in dataset_config.zarr_positions:
            manual_tps = set(
                get_annotated_timepoints_for_position(dataset_config, position, manual_annotations)
            )

            auto_tps = set(
                get_annotated_timepoints_for_position(dataset_config, position, auto_annotations)
            )

            list_of_missed_tps = list(manual_tps - auto_tps) if manual_tps - auto_tps else np.NaN

            stats.append(
                {
                    "dataset_name": dataset_name,
                    "position": position,
                    "n_auto_detected": len(auto_tps),
                    "n_manual_annotated": len(manual_tps),
                    "n_missed": len(manual_tps - auto_tps),
                    "list_of_missed_tps": list_of_missed_tps,
                    "n_tps_assessed": dataset_config.duration,
                }
            )

    df = pd.DataFrame(stats)

    total_manual = df["n_manual_annotated"].sum()
    total_auto = df["n_auto_detected"].sum()
    total_missed = df["n_missed"].sum()
    percent_missed = (total_missed / total_manual) * 100 if total_manual > 0 else 0
    total_timepoints = df["n_tps_assessed"].sum()
    percent_artifact = (total_auto + total_missed) / total_timepoints * 100

    results = (
        f"--- {annotation_type} STATISTICS ---\n"
        f"Total manual annotated timepoints: {total_manual}\n"
        f"Total missed timepoints: {total_missed}\n"
        f"Percent of missed timepoints: {percent_missed:.3f}%\n"
        f"Total auto-detected timepoints: {total_auto}\n"
        f"Total timepoints assessed: {total_timepoints}\n"
        f"Percent of tps with artifacts: {percent_artifact:.3f}%"
    )

    return results
