# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.endo_pipeline.configs import get_available_zarr_files, load_dataset_config
from src.endo_pipeline.io.input import load_zarr_as_dask_array

# %% SET PARAMS
LOWER_THRESH = 0.005  # Percentage to use for thresholding
UPPER_THRESH = 0.01  # Percentage to use for thresholding
ROLLING_WINDOW = 100  # Size of the rolling window for mean calculation (4 timepoints)
NUM_ZSLICES = 25  # Number of z-slices per timepoint

# %% LOAD DATA
dataset_name = "20250224_20X"
# dataset_name = "20250618_20X"
dataset_config = load_dataset_config(dataset_name)

zarr_files = get_available_zarr_files(dataset_config)

for position in dataset_config.zarr_positions:
    bf_zarr = load_zarr_as_dask_array(zarr_files[position], channels=["BF"], level=1)

    # FIND OUTLIERS
    bf_zarr.squeeze()  # bf_zarr shape = (timepoints, z, x, y)

    # 1 Compute mean intensity over x/y axes on dask array
    intensity_array = bf_zarr.mean(axis=(-2, -1))
    flattened_img_data = intensity_array.flatten()

    # 3 Convert to pandas Series for rolling median
    data_np = flattened_img_data.compute()
    series = pd.Series(data_np)

    rolling_median = series.rolling(ROLLING_WINDOW, center=True).median()

    # Pad edges with median of first/last window
    start_pad_value = np.median(data_np[:ROLLING_WINDOW])
    end_pad_value = np.median(data_np[-ROLLING_WINDOW:])

    rolling_median.iloc[: ROLLING_WINDOW // 2] = start_pad_value
    rolling_median.iloc[-ROLLING_WINDOW // 2 :] = end_pad_value

    rolling_median_np = rolling_median.to_numpy()

    # Set thresholds
    dark_threshold = rolling_median_np * (1 - UPPER_THRESH)
    partial_dark_threshold = rolling_median_np * (1 - LOWER_THRESH)
    bright_threshold = rolling_median_np * (1 + UPPER_THRESH)

    # Find local minima (dark outliers) and maxima (bright outliers)
    minima, _ = find_peaks(-data_np)  # dark
    maxima, _ = find_peaks(data_np)  # bright

    # Keep only points beyond thresholds
    dark_outliers = [i for i in minima if data_np[i] <= dark_threshold[i]]
    partial_dark_outliers = [
        i for i in minima if data_np[i] <= partial_dark_threshold[i] and i not in dark_outliers
    ]
    bright_outliers = [i for i in maxima if data_np[i] >= bright_threshold[i]]

    # Separate dictionaries
    dark_outlier_dict = {}
    partial_outlier_dict = {}
    bright_outlier_dict = {}

    # Populate dark outliers
    for idx in dark_outliers:
        t = idx // NUM_ZSLICES
        z = idx % NUM_ZSLICES
        dark_outlier_dict.setdefault(t, []).append(z)

    # Populate partial dark outliers
    for idx in partial_dark_outliers:
        t = idx // NUM_ZSLICES
        z = idx % NUM_ZSLICES
        partial_outlier_dict.setdefault(t, []).append(z)

    # Populate bright outliers
    for idx in bright_outliers:
        t = idx // NUM_ZSLICES
        z = idx % NUM_ZSLICES
        bright_outlier_dict.setdefault(t, []).append(z)

    # Print results per position
    print(f"Position {position} Dark Outliers (timepoint: [z-slice])")
    for tp, z_list in dark_outlier_dict.items():
        print(f"{tp}: {z_list}")

    print(f"Position {position} Partial Dark Outliers (timepoint: [z-slice])")
    for tp, z_list in partial_outlier_dict.items():
        print(f"{tp}: {z_list}")

    print(f"Position {position} Bright Outliers (timepoint: [z-slice])")
    for tp, z_list in bright_outlier_dict.items():
        print(f"{tp}: {z_list}")

    # Plot
    plt.figure(figsize=(12, 10))
    plt.plot(data_np, label="Intensity", color="black", alpha=0.5)
    plt.plot(rolling_median_np, label="Rolling Mean", color="gray", alpha=0.5)
    plt.plot(dark_threshold, label="Lower Threshold", color="red", linestyle="--")
    plt.plot(partial_dark_threshold, label="Partial Dark Threshold", color="purple", linestyle="--")
    plt.plot(bright_threshold, label="Upper Threshold", color="orange", linestyle="--")
    plt.scatter(dark_outliers, data_np[dark_outliers], color="red", label="Dark Outliers", zorder=5)
    plt.scatter(
        partial_dark_outliers,
        data_np[partial_dark_outliers],
        color="purple",
        label="Partial Dark Outliers",
        zorder=5,
    )
    plt.scatter(
        bright_outliers, data_np[bright_outliers], color="orange", label="Bright Outliers", zorder=5
    )
    plt.xlabel("Flattened Index")
    plt.ylabel("Intensity")
    plt.title(f"{dataset_name} - Position {position}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
