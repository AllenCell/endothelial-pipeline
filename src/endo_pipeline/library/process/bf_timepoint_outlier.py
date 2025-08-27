import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from endo_pipeline.configs import DatasetConfig, get_available_zarr_files
from endo_pipeline.io.input import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path, save_plot_to_path

THRESHOLD1 = 0.004  # Percentage to use for thresholding
THRESHOLD2 = 0.01  # Percentage to use for thresholding
ROLLING_WINDOW = 100  # Size of the rolling window for mean calculation (4 timepoints)
NUM_ZSLICES = 25  # Number of z-slices per timepoint


def plot_outliers(
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
    """Plot intensity data with thresholds and outliers, embedding outlier info on the plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(data_np, label="Intensity", color="black", alpha=0.5)
    ax.plot(rolling_median_np, label="Rolling Median", color="black", alpha=1, zorder=4)
    ax.plot(dark_threshold, label="Lower Threshold", color="red", linestyle="--")
    ax.plot(partial_dark_threshold, label="Partial Dark Threshold", color="purple", linestyle="--")
    ax.plot(bright_threshold, label="Upper Threshold", color="orange", linestyle="--")

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
            for t in set(i // num_zslices for i in indices)
        }
        if d:
            info_lines.append(f"{title}:\n" + "\n".join(f"{t}: {z}" for t, z in d.items()))

    # Add to plot
    if len(info_lines) > 1:
        fig.text(
            1.02,
            0.5,
            "\n\n".join(info_lines),
            fontsize=10,
            va="center",
            ha="left",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Flattened Index")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{dataset_name} - Position {position}")
    ax.legend()
    fig.tight_layout(rect=[0, 0, 0.8, 1])  # leave space on right
    plt.show()

    save_dir = get_output_path("brightfield_outliers", dataset_name)
    save_plot_to_path(fig, save_dir, f"bf_outliers_P{position}")


def detect_outliers(dataset_config: DatasetConfig, position: int, visualize: bool = False):
    zarr_files = get_available_zarr_files(dataset_config)
    bf_zarr = load_zarr_as_dask_array(zarr_files[position], channels=["BF"], level=1)
    bf_zarr.squeeze()  # shape = (timepoints, z, x, y)

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

    # --- Return values ---
    bf_scope_error = sorted({idx // NUM_ZSLICES for idx in partial_dark_outliers})
    bf_temp_artifact = sorted({idx // NUM_ZSLICES for idx in (dark_outliers + bright_outliers)})

    # Summary counts
    print(f"Summary for Position {position}:")
    print(f"BF Scope Errors : {len(bf_scope_error)} timepoints")
    print(f"BF Temp Artifacts: {len(bf_temp_artifact)} timepoints")

    # --- Visualization ---
    if visualize:
        plot_outliers(
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
