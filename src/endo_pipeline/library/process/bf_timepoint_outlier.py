import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    get_available_zarr_files,
    load_dataset_config,
)
from endo_pipeline.io.input import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.settings.image_data import NUM_ZSLICES

THRESHOLD1 = 0.004
"""Percentage to use for thresholding partial dark outliers."""

THRESHOLD2 = 0.01
"""Percentage to use for thresholding dark and bright outliers."""

ROLLING_WINDOW = 100
"""Number of z-slices per to use for rolling window calculation (4 timepoints)."""


def plot_bf_outliers(
    data_np: np.ndarray,
    rolling_median_np: np.ndarray,
    dark_threshold: np.ndarray,
    partial_dark_threshold: np.ndarray,
    bright_threshold: np.ndarray,
    dark_outliers: list[int],
    partial_dark_outliers: list[int],
    bright_outliers: list[int],
    dataset_name: str,
    position: int,
    num_zslices: int = NUM_ZSLICES,
) -> None:
    """
    Plot intensity data with thresholds and outliers, embedding outlier info on the plot.

    Parameters
    ----------
    data_np:
        The intensity data as a 1D numpy array, representing flattened indices of timepoints and z-slices.
    rolling_median_np:
        The rolling median of the intensity data, used as a baseline for comparison.
    dark_threshold:
        The lower threshold for detecting dark outliers.
    partial_dark_threshold:
        The partial dark threshold for detecting less severe dark outliers.
    bright_threshold:
        The upper threshold for detecting bright outliers.
    dark_outliers:
        Indices of dark outliers in the data.
    partial_dark_outliers:
        Indices of partial dark outliers in the data.
    bright_outliers:
        Indices of bright outliers in the data.
    dataset_name:
        The name of the dataset being analyzed, used for labeling the plot.
    position:
        The position identifier within the dataset, used for labeling the plot.
    num_zslices:
        The number of z-slices per timepoint (default is NUM_ZSLICES).
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.plot(data_np, label="Intensity", color="black", alpha=0.5)
    ax.plot(
        rolling_median_np,
        label="Rolling Median (window = 100 z-slices (4 tps))",
        color="black",
        alpha=1,
        zorder=4,
    )
    ax.plot(
        dark_threshold, label=f"Lower Threshold ({THRESHOLD2*100}%)", color="red", linestyle="--"
    )
    ax.plot(
        partial_dark_threshold,
        label=f"Partial Dark Threshold ({THRESHOLD1*100}%)",
        color="purple",
        linestyle="--",
    )
    ax.plot(
        bright_threshold,
        label=f"Upper Threshold ({THRESHOLD2*100}%)",
        color="orange",
        linestyle="--",
    )

    ax.scatter(dark_outliers, data_np[dark_outliers], color="red", label="Dark Outliers", zorder=5)
    ax.scatter(
        partial_dark_outliers,
        data_np[partial_dark_outliers],
        color="purple",
        label="Partial Dark Outliers",
        zorder=5,
    )
    ax.scatter(
        bright_outliers, data_np[bright_outliers], color="orange", label="Bright Outliers", zorder=5
    )

    outlier_groups = [
        ("Dark", dark_outliers),
        ("Partial Dark", partial_dark_outliers),
        ("Bright", bright_outliers),
    ]

    info_lines = ["timepoint: [z-slices]\n"]
    for title, indices in outlier_groups:
        d = {
            t: [i % num_zslices for i in indices if i // num_zslices == t]
            for t in sorted(set(i // num_zslices for i in indices))
        }
        if d:
            info_lines.append(f"{title}:\n" + "\n".join(f"{t}: {z}" for t, z in d.items()))

    if len(info_lines) > 1:
        fig.text(
            1.02,
            0.5,
            "\n\n".join(info_lines),
            fontsize=12,
            va="center",
            ha="left",
            transform=ax.transAxes,
        )

    mean_for_lim = np.mean(data_np)
    ax.set_xlabel("Index (flatted Z-slices)", fontsize=16, labelpad=12)
    ax.set_ylabel("Average bright-feild intensity in Z-slice (a.u.)", fontsize=16, labelpad=12)

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=12)

    # Secondary X-axis for timepoints
    def index_to_tp(x):
        return x // num_zslices

    def tp_to_index(t):
        return t * num_zslices

    secax = ax.secondary_xaxis("top", functions=(index_to_tp, tp_to_index))
    secax.set_xlabel("Time (frames)", fontsize=16, labelpad=12)
    max_tp = data_np.shape[0] // num_zslices
    secax.set_xticks(np.arange(0, max_tp + 1, 25))
    secax.tick_params(axis="x", labelsize=14)

    ax.legend(fontsize=14, loc="upper right", frameon=True)

    ax.set_ylim(mean_for_lim - mean_for_lim * 0.05, mean_for_lim + mean_for_lim * 0.05)

    fig.tight_layout(rect=[0, 0, 0.8, 1])

    save_dir = get_output_path("single_timepoint_outlier_detection", dataset_name)
    save_plot_to_path(fig, save_dir, f"bf_outliers_{dataset_name}_P{position}", file_format=".pdf")
    plt.show()
    plt.close(fig)


def detect_bf_outliers(
    dataset_config: DatasetConfig, position: int, visualize: bool = False
) -> tuple[list[int], list[int]]:
    """
    Detect outliers in brightfield (BF) microscopy data based on intensity thresholds.

    Parameters
    ----------
    dataset_config:
        Configuration object containing metadata and paths for the dataset.
    position:
        The position index within the dataset to analyze.
    visualize:
        If True, generates and saves plots of the intensity data, thresholds, and outliers.

    Returns
    -------
    A tuple containing two lists:
    - `bf_scope_error`: Sorted list of timepoints with partial dark outliers.
    - `bf_temp_artifact`: Sorted list of timepoints with dark or bright outliers.
    """
    zarr_files = get_available_zarr_files(dataset_config)
    bf_zarr = load_zarr_as_dask_array(zarr_files[position], channels=["BF"], level=1, squeeze=True)

    # 1 Compute mean intensity over x/y axes
    intensity_array = bf_zarr.mean(axis=(-2, -1))
    flattened_img_data = intensity_array.flatten()

    # 2 Convert to pandas Series for rolling median
    data_np = flattened_img_data.compute()
    series = pd.Series(data_np)
    rolling_median = series.rolling(ROLLING_WINDOW, center=True).median()

    # Pad edges
    start_pad_value = np.median(data_np[:ROLLING_WINDOW])
    end_pad_value = np.median(data_np[-ROLLING_WINDOW:])
    rolling_median.iloc[: ROLLING_WINDOW // 2] = start_pad_value
    rolling_median.iloc[-ROLLING_WINDOW // 2 :] = end_pad_value
    rolling_median_np = rolling_median.to_numpy()

    # Thresholds
    dark_threshold = rolling_median_np * (1 - THRESHOLD2)
    partial_dark_threshold = rolling_median_np * (1 - THRESHOLD1)
    bright_threshold = rolling_median_np * (1 + THRESHOLD2)

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

    if visualize:
        plot_bf_outliers(
            data_np,
            rolling_median_np,
            dark_threshold,
            partial_dark_threshold,
            bright_threshold,
            dark_outliers,
            partial_dark_outliers,
            bright_outliers,
            dataset_config.name,
            position,
        )

    return bf_scope_error, bf_temp_artifact


def performance_stats(
    datasets: list[str],
    manual_annotations: list[TimepointAnnotation],
    auto_annotations: list[TimepointAnnotation],
    annotation_type: str,
) -> None:
    """
    Calculate and print statistics for manual and automatic timepoint annotations.

    This function processes a list of datasets, compares manual and automatic annotations
    for each dataset and position, and calculates statistics such as the number of missed
    timepoints, total annotated timepoints, and artifact percentages. The results are printed
    to the console.

    Args:
        datasets (list[str]):
            A list of dataset names to process.
        manual_annotations (list[TimepointAnnotation]):
            A list of manual annotation types to consider.
        auto_annotations (list[TimepointAnnotation]):
            A list of automatic annotation types to consider.
        annotation_type (str):
            A string indicating the type of annotation (e.g., "Brightfield" or "GFP")
            for labeling the output statistics.
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

    print(f"--- {annotation_type} STATISTICS ---")
    print(f"Total manual annotated timepoints: {total_manual}")
    print(f"Total missed timepoints: {total_missed}")
    print(f"Percent of missed timepoints: {percent_missed:.2f}%")
    print(f"Total auto-detected timepoints: {total_auto}")
    print(f"Total timepoints assessed: {total_timepoints}")
    print(f"Percent of tps with artifacts: {percent_artifact:.2f}%")
    print()
