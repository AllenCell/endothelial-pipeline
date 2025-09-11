# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
from endo_pipeline.io import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.library.process.z_stack_selection import (
    plot_histogram_upper_slices_available,
    plot_vlines,
    visualize_z_slices_with_offsets,
)
from endo_pipeline.settings import LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET

# %%
TIMEPOINT = 0

datasets = [
    # "20241120_20X", # To be excluded
    # "20241217_20X", # To be excluded
    "20250224_20X",
    "20250319_20X",
    "20250326_20X",
    "20250331_20X",
    "20250402_20X",
    "20250409_20X",
    "20250428_20X",
    "20250604_20X",
    "20250611_20X",
    "20250618_20X",
    "20250714_20X",
    "20250716_20X",
    "20250728_20X",
    "20250806_20X",
    "20250813_20X",
    "20250818_20X",
    "20250825_20X",
    "20250827_20X",
]

# %%
save_dir = get_output_path(
    "z_range_selection", f"offsets_{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}", "images"
)
for dataset in datasets:
    dataset_config = load_dataset_config(dataset)
    for position in dataset_config.zarr_positions:
        visualize_z_slices_with_offsets(dataset_config, position, TIMEPOINT, save_dir)
        break

# %%
save_dir = get_output_path("z_range_selection")
plot_histogram_upper_slices_available(datasets, save_dir)

# %%
colormap = colormaps["tab20"]
colors = [colormap(i / len(datasets)) for i in range(len(datasets))]

for POSITION in range(6):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, dataset in enumerate(datasets):
        dataset_config = load_dataset_config(dataset)
        save_dir = get_output_path(
            "z_range_selection", f"offsets_{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}"
        )
        zarr_file = get_zarr_file_for_position(dataset_config, POSITION)

        bf_stack = load_zarr_as_dask_array(
            zarr_file, channels=["BF"], timepoints=TIMEPOINT, level=1, squeeze=True
        )
        cdh5_stack = load_zarr_as_dask_array(
            zarr_file, channels=["EGFP"], timepoints=TIMEPOINT, level=1, squeeze=True
        )

        if dataset_config.center_z_plane is None:
            print(f"The center slice is None for dataset {dataset_config.name}")
            continue
        else:
            center_slice = dataset_config.center_z_plane[POSITION]

        top_slice = 24
        available_slices_above = top_slice - center_slice

        if UPPER_Z_SLICE_OFFSET > available_slices_above:
            print(f"Not enough slices above center for dataset {dataset_config.name}, skipping...")
            continue

        # Calculate CDH5 histogram
        cdh5_hist = np.array(
            [np.sum(cdh5_stack[z, :, :].compute()) for z in range(cdh5_stack.shape[0])]
        )
        # Calculate the standard deviation for each Z slice in the Brightfield stack
        bf_std = np.array([np.std(bf_stack[z, :, :].compute()) for z in range(bf_stack.shape[0])])

        # Normalize the x-axis by subtracting the center slice
        normalized_x = np.arange(len(bf_std)) - center_slice

        # Normalize bf_std and cdh5_hist by their respective maximum values
        # bf_std_normalized = bf_std / np.min(bf_std) if np.min(bf_std) != 0 else bf_std
        bf_std_normalized = bf_std / bf_std[center_slice] if bf_std[center_slice] != 0 else bf_std
        cdh5_hist_normalized = (
            cdh5_hist / np.max(cdh5_hist) if np.max(cdh5_hist) != 0 else cdh5_hist
        )
        # cdh5_hist_normalized = cdh5_hist / cdh5_hist[center_slice] if cdh5_hist[center_slice] != 0 else cdh5_hist

        # Plot Brightfield Standard Deviation with normalized x-axis
        axes[0].plot(normalized_x, bf_std_normalized, color=colors[i])
        axes[0].set_xlabel("Normalized Z Slice (Center = 0)")
        axes[0].set_ylabel("Normalized BF Standard Deviation")
        axes[0].set_ylim(0.99, 1.35)

        # Plot CDH5 Total Intensity
        axes[1].plot(normalized_x, cdh5_hist_normalized, label=dataset_config.name, color=colors[i])
        axes[1].set_xlabel("Normalized Z Slice (Center = 0)")
        axes[1].set_ylabel("Normalized CDH5 Total Intensity")
        axes[1].set_ylim(0.84, 1.1)

    plot_vlines(
        axes[0],
        0,
        LOWER_Z_SLICE_OFFSET,
        UPPER_Z_SLICE_OFFSET,
        y_min=axes[0].get_ylim()[0],
        y_max=axes[0].get_ylim()[1],
    )
    plot_vlines(
        axes[1],
        0,
        LOWER_Z_SLICE_OFFSET,
        UPPER_Z_SLICE_OFFSET,
        y_min=axes[1].get_ylim()[0],
        y_max=axes[1].get_ylim()[1],
    )
    axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)
    plt.suptitle(
        f"Position {POSITION} Timepoint {TIMEPOINT}, Offset Range: -{LOWER_Z_SLICE_OFFSET} to +{UPPER_Z_SLICE_OFFSET}"
    )
    plt.show()
    save_dir = get_output_path(
        "z_range_selection",
        f"offsets_{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}",
        "normalized_profiles",
    )
    save_plot_to_path(fig, save_dir, f"pos{POSITION}_tp{TIMEPOINT}_normalized_profiles")

# %%
for dataset in ["20250813_20X"]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dataset_config = load_dataset_config(dataset)
    for POSITION in dataset_config.zarr_positions:
        center_slice = dataset_config.center_z_plane[POSITION]

        save_dir = get_output_path(
            "z_range_selection", f"offsets_{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}"
        )
        zarr_file = get_zarr_file_for_position(dataset_config, POSITION)

        # for TIMEPOINT in [0, 90, 180, 270]:
        bf_stack = load_zarr_as_dask_array(
            zarr_file, channels=["BF"], timepoints=TIMEPOINT, level=1, squeeze=True
        )
        cdh5_stack = load_zarr_as_dask_array(
            zarr_file, channels=["EGFP"], timepoints=TIMEPOINT, level=1, squeeze=True
        )

        # Calculate CDH5 histogram
        cdh5_hist = np.array(
            [np.sum(cdh5_stack[z, :, :].compute()) for z in range(cdh5_stack.shape[0])]
        )
        # Calculate the standard deviation for each Z slice in the Brightfield stack
        bf_std = np.array([np.std(bf_stack[z, :, :].compute()) for z in range(bf_stack.shape[0])])

        # Normalize the x-axis by subtracting the center slice
        normalized_x = np.arange(len(bf_std)) - center_slice

        # Normalize bf_std and cdh5_hist by their respective maximum values
        # bf_std_normalized = bf_std / np.min(bf_std) if np.min(bf_std) != 0 else bf_std
        bf_std_normalized = bf_std / bf_std[center_slice] if bf_std[center_slice] != 0 else bf_std
        cdh5_hist_normalized = (
            cdh5_hist / np.max(cdh5_hist) if np.max(cdh5_hist) != 0 else cdh5_hist
        )
        # cdh5_hist_normalized = cdh5_hist / cdh5_hist[center_slice] if cdh5_hist[center_slice] != 0 else cdh5_hist

        # Plot Brightfield Standard Deviation with normalized x-axis
        axes[0].plot(normalized_x, bf_std_normalized)
        axes[0].set_xlabel("Normalized Z Slice (Center = 0)")
        axes[0].set_ylabel("Normalized BF Standard Deviation")
        axes[0].set_ylim(0.99, 1.35)

        # Plot CDH5 Total Intensity
        axes[1].plot(
            normalized_x, cdh5_hist_normalized, label=f"{dataset_config.name}, P{POSITION}"
        )
        axes[1].set_xlabel("Normalized Z Slice (Center = 0)")
        axes[1].set_ylabel("Normalized CDH5 Total Intensity")
        axes[1].set_ylim(0.84, 1.1)

    plot_vlines(
        axes[0],
        0,
        LOWER_Z_SLICE_OFFSET,
        UPPER_Z_SLICE_OFFSET,
        y_min=axes[0].get_ylim()[0],
        y_max=axes[0].get_ylim()[1],
    )
    plot_vlines(
        axes[1],
        0,
        LOWER_Z_SLICE_OFFSET,
        UPPER_Z_SLICE_OFFSET,
        y_min=axes[1].get_ylim()[0],
        y_max=axes[1].get_ylim()[1],
    )
    axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)
    plt.suptitle(
        f"{dataset_config.name} TP {TIMEPOINT}, Offset Range: -{LOWER_Z_SLICE_OFFSET} to +{UPPER_Z_SLICE_OFFSET}"
    )
    plt.show()
    save_dir = get_output_path(
        "z_range_selection",
        f"offsets_{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}",
        "normalized_profiles",
    )
    save_plot_to_path(fig, save_dir, f"{dataset_config.name}_tp{TIMEPOINT}_normalized_profiles")
# %%
