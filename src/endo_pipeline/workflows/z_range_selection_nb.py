# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
from endo_pipeline.io import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.library.process.image_processing import contrast_stretching

# %%
TIMEPOINT = 90
Z_SLICE_LOWER_OFFSET = 4
Z_SLICE_UPPER_OFFSET = 11


# %%
def plot_vlines(
    axis: plt.Axes,
    center: int,
    lower_offset: int,
    upper_offset: int,
    y_min: float,
    y_max: float,
) -> None:
    """Plot vertical lines of global center and offsets."""
    axis.vlines(
        center,
        ymin=y_min,
        ymax=y_max,
        colors="red",
        linestyles="solid",
        label=f"Global Center Slice {center}",
    )
    axis.vlines(
        center - lower_offset,
        ymin=y_min,
        ymax=y_max,
        colors="magenta",
        linestyles="dashed",
        label=f"Center - {lower_offset} Slices {center - lower_offset}",
    )
    axis.vlines(
        center + upper_offset,
        ymin=y_min,
        ymax=y_max,
        colors="black",
        linestyles="dashed",
        label=f"Center + {upper_offset} Slices {center + upper_offset}",
    )


# %%
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
]

# %%
# for dataset in [datasets[0]]:
for dataset in datasets:
    dataset_config = load_dataset_config(dataset)
    for position in dataset_config.zarr_positions:
        zarr_file = get_zarr_file_for_position(dataset_config, position)

        bf_stack = load_zarr_as_dask_array(
            zarr_file, channels=["BF"], timepoints=TIMEPOINT, level=1, squeeze=True
        )
        cdh5_stack = load_zarr_as_dask_array(
            zarr_file, channels=["EGFP"], timepoints=TIMEPOINT, level=1, squeeze=True
        )

        if dataset_config.center_z_plane is None:
            # Handle the case where the value is None
            print(f"The center slice is None for dataset {dataset_config.name}")
            continue
        else:
            center_slice = dataset_config.center_z_plane[position]

        top_slice = 24
        available_slices_above = top_slice - center_slice

        if Z_SLICE_UPPER_OFFSET > available_slices_above:
            print(f"Not enough slices above center for dataset {dataset_config.name}, skipping...")
            continue

        # Brightfield (bf) variables
        bf_center = bf_stack[center_slice, :, :].compute()
        bf_top = bf_stack[top_slice, :, :].compute()
        bf_lower_offset = bf_stack[center_slice - Z_SLICE_LOWER_OFFSET, :, :].compute()
        bf_upper_offset = bf_stack[center_slice + Z_SLICE_UPPER_OFFSET, :, :].compute()

        # CDH5 (cdh5) variables
        cdh5_center = cdh5_stack[center_slice, :, :].compute()
        cdh5_top = cdh5_stack[top_slice, :, :].compute()
        cdh5_lower_offset = cdh5_stack[center_slice - Z_SLICE_LOWER_OFFSET, :, :].compute()
        cdh5_upper_offset = cdh5_stack[center_slice + Z_SLICE_UPPER_OFFSET, :, :].compute()

        # Brightfield (bf) min and max calculations
        min_bf = np.percentile(bf_center, 0.2)
        max_bf = np.percentile(bf_center, 99.8)

        # CDH5 (cdh5) min and max calculations
        min_cdh5 = np.percentile(cdh5_center, 0.2)
        max_cdh5 = np.percentile(cdh5_center, 99.8)

        # Contrast stretching
        bf_center = contrast_stretching(bf_center, custom_range=(min_bf, max_bf))
        bf_top = contrast_stretching(bf_top, custom_range=(min_bf, max_bf))
        bf_lower_offset = contrast_stretching(bf_lower_offset, custom_range=(min_bf, max_bf))
        bf_upper_offset = contrast_stretching(bf_upper_offset, custom_range=(min_bf, max_bf))

        cdh5_center = contrast_stretching(cdh5_center, custom_range=(min_cdh5, max_cdh5))
        cdh5_top = contrast_stretching(cdh5_top, custom_range=(min_cdh5, max_cdh5))
        cdh5_lower_offset = contrast_stretching(
            cdh5_lower_offset, custom_range=(min_cdh5, max_cdh5)
        )
        cdh5_upper_offset = contrast_stretching(
            cdh5_upper_offset, custom_range=(min_cdh5, max_cdh5)
        )

        # Define the data and titles for each subplot
        bf_slices = [bf_lower_offset, bf_center, bf_upper_offset, bf_top]
        cdh5_slices = [cdh5_lower_offset, cdh5_center, cdh5_upper_offset, cdh5_top]
        titles = [
            f"Lower Offset Slice - {center_slice - Z_SLICE_LOWER_OFFSET}",
            f"Center Slice - {center_slice}",
            f"Upper Offset Slice - {center_slice + Z_SLICE_UPPER_OFFSET}",
            "Top Slice - 24",
        ]

        fig, axes = plt.subplots(2, 4, figsize=(22, 12))  # 2 rows, 4 columns
        for i in range(4):
            # Brightfield (BF) slices
            axes[0, i].imshow(bf_slices[i], cmap="gray")
            axes[0, i].set_title(f"BF {titles[i]}")
            axes[0, i].axis("off")

            # CDH5 slices
            axes[1, i].imshow(cdh5_slices[i], cmap="gray")
            axes[1, i].set_title(f"CDH5 {titles[i]}")
            axes[1, i].axis("off")

        plt.suptitle(f"{dataset_config.name} Position {position} Timepoint {TIMEPOINT}\n")
        plt.tight_layout()
        plt.show()
        save_dir = get_output_path(
            "z_range_selection", f"offsets_{Z_SLICE_LOWER_OFFSET}_{Z_SLICE_UPPER_OFFSET}", "images"
        )
        save_plot_to_path(
            fig, save_dir, f"{dataset_config.name}_pos{position}_tp{TIMEPOINT}_im_slices"
        )
        plt.close()

# %%
data = []

for dataset in datasets:
    dataset_config = load_dataset_config(dataset)
    for position in dataset_config.zarr_positions:
        center_slice = dataset_config.center_z_plane[position]
        top_slice = 24
        available_slices_above = top_slice - center_slice
        data.append(
            {
                "dataset": dataset_config.name,
                "position": position,
                "available_slices_above": available_slices_above,
            }
        )

df = pd.DataFrame(data)

fig = plt.figure(figsize=(6, 6))
plt.hist(df["available_slices_above"], bins=range(8, 20, 1), align="left", edgecolor="black")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Available Slices Above Center Slice")
plt.ylabel("Number of Positions")

df_11 = df[df["available_slices_above"] == 11]
limiting_datasets = df_11.dataset.unique()
text = "Datasets in 11:\n" + "\n".join(limiting_datasets)
plt.text(
    0.95,
    0.95,
    text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(facecolor="white", alpha=0.8),
)

plt.show()
save_dir = get_output_path("z_range_selection")
save_plot_to_path(fig, save_dir, "available_slices_above_center_histogram")

# %%
colormap = colormaps["tab20"]
colors = [colormap(i / len(datasets)) for i in range(len(datasets))]

for POSITION in range(6):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, dataset in enumerate(datasets):
        dataset_config = load_dataset_config(dataset)
        save_dir = get_output_path(
            "z_range_selection", f"offsets_{Z_SLICE_LOWER_OFFSET}_{Z_SLICE_UPPER_OFFSET}"
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

        if Z_SLICE_UPPER_OFFSET > available_slices_above:
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

        # Plot CDH5 Total Intensity
        axes[1].plot(normalized_x, cdh5_hist_normalized, label=dataset_config.name, color=colors[i])
        axes[1].set_xlabel("Normalized Z Slice (Center = 0)")
        axes[1].set_ylabel("Normalized CDH5 Total Intensity")

    plot_vlines(
        axes[0],
        0,
        Z_SLICE_LOWER_OFFSET,
        Z_SLICE_UPPER_OFFSET,
        y_min=axes[0].get_ylim()[0],
        y_max=axes[0].get_ylim()[1],
    )
    plot_vlines(
        axes[1],
        0,
        Z_SLICE_LOWER_OFFSET,
        Z_SLICE_UPPER_OFFSET,
        y_min=axes[1].get_ylim()[0],
        y_max=axes[1].get_ylim()[1],
    )
    axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)
    plt.suptitle(
        f"Position {POSITION} Timepoint {TIMEPOINT}, Offset Range: -{Z_SLICE_LOWER_OFFSET} to +{Z_SLICE_UPPER_OFFSET}"
    )
    plt.show()
    save_dir = get_output_path(
        "z_range_selection",
        f"offsets_{Z_SLICE_LOWER_OFFSET}_{Z_SLICE_UPPER_OFFSET}",
        "normalized_profiles",
    )
    save_plot_to_path(fig, save_dir, f"pos{POSITION}_tp{TIMEPOINT}_normalized_profiles")

# %%
for dataset in datasets:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dataset_config = load_dataset_config(dataset)
    for POSITION in dataset_config.zarr_positions:
        center_slice = dataset_config.center_z_plane[POSITION]

        save_dir = get_output_path(
            "z_range_selection", f"offsets_{Z_SLICE_LOWER_OFFSET}_{Z_SLICE_UPPER_OFFSET}"
        )
        zarr_file = get_zarr_file_for_position(dataset_config, POSITION)

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

        # Plot CDH5 Total Intensity
        axes[1].plot(
            normalized_x, cdh5_hist_normalized, label=f"{dataset_config.name}, P{POSITION}"
        )
        axes[1].set_xlabel("Normalized Z Slice (Center = 0)")
        axes[1].set_ylabel("Normalized CDH5 Total Intensity")

    plot_vlines(
        axes[0],
        0,
        Z_SLICE_LOWER_OFFSET,
        Z_SLICE_UPPER_OFFSET,
        y_min=axes[0].get_ylim()[0],
        y_max=axes[0].get_ylim()[1],
    )
    plot_vlines(
        axes[1],
        0,
        Z_SLICE_LOWER_OFFSET,
        Z_SLICE_UPPER_OFFSET,
        y_min=axes[1].get_ylim()[0],
        y_max=axes[1].get_ylim()[1],
    )
    axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)
    plt.suptitle(
        f"{dataset_config.name} TP {TIMEPOINT}, Offset Range: -{Z_SLICE_LOWER_OFFSET} to +{Z_SLICE_UPPER_OFFSET}"
    )
    plt.show()
    save_dir = get_output_path(
        "z_range_selection",
        f"offsets_{Z_SLICE_LOWER_OFFSET}_{Z_SLICE_UPPER_OFFSET}",
        "normalized_profiles",
    )
    save_plot_to_path(fig, save_dir, f"{dataset_config.name}_tp{TIMEPOINT}_normalized_profiles")
# %%
