# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from endo_pipeline.io import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.library.process.image_processing import contrast_stretching
from endo_pipeline.library.process.z_stack_selection import (
    append_projection_outputs,
    get_center_plane_for_position,
    plot_bottom_top_slices,
    plot_image_row,
    save_projection_image,
)
from endo_pipeline.library.visualize.model_inputs.image_processing_steps import (
    process_brightfield,
    process_cdh5,
)
from endo_pipeline.library.visualize.model_inputs.plot import visualize_images_with_histograms

# %%
POSITION = 0
TIMEPOINT = 250
Z_SLICE_LOWER_OFFSET = 4
Z_SLICE_UPPER_OFFSET = 11
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

availabe_slices_above_list = []
dataset_list = []

for dataset in datasets:
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
        # Handle the case where the value is None
        print(f"The center slice is None for dataset {dataset_config.name}")
        continue
    else:
        center_slice = dataset_config.center_z_plane[POSITION]

    top_slice = 24
    available_slices_above = top_slice - center_slice
    availabe_slices_above_list.append(available_slices_above)

    if Z_SLICE_UPPER_OFFSET > available_slices_above:
        print(f"Not enough slices above center for dataset {dataset_config.name}, skipping...")
        continue

    # Helper function to plot vertical lines
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

    # Calculate CDH5 histogram
    cdh5_hist = np.array(
        [np.sum(cdh5_stack[z, :, :].compute()) for z in range(cdh5_stack.shape[0])]
    )
    # Calculate the standard deviation for each Z slice in the Brightfield stack
    bf_std = np.array([np.std(bf_stack[z, :, :].compute()) for z in range(bf_stack.shape[0])])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Brightfield Standard Deviation
    axes[0].plot(bf_std, label="BF Std Dev", color="blue")
    axes[0].set_xlabel("Z Slice")
    axes[0].set_ylabel("BF Standard Deviation")
    plot_vlines(
        axes[0],
        center_slice,
        Z_SLICE_LOWER_OFFSET,
        Z_SLICE_UPPER_OFFSET,
        np.min(bf_std),
        np.max(bf_std),
    )

    # Plot CDH5 Total Intensity
    axes[1].plot(cdh5_hist, color="green")
    axes[1].set_xlabel("Z Slice")
    axes[1].set_ylabel("CDH5 Total Intensity")
    plot_vlines(
        axes[1],
        center_slice,
        Z_SLICE_LOWER_OFFSET,
        Z_SLICE_UPPER_OFFSET,
        np.min(cdh5_hist),
        np.max(cdh5_hist),
    )

    # Adjust legend and layout
    axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)
    plt.suptitle(
        f"{dataset_config.name} Position {POSITION} Timepoint {TIMEPOINT}, Available Slices Above Center: {available_slices_above}"
    )
    plt.tight_layout()
    plt.show()
    save_plot_to_path(fig, save_dir, f"{dataset_config.name}_pos{POSITION}_tp{TIMEPOINT}_zprofile")
    plt.close()

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
    min_bf = np.min(
        [
            np.percentile(bf_center, 0.2),
            np.percentile(bf_top, 0.2),
            np.percentile(bf_lower_offset, 0.2),
            np.percentile(bf_upper_offset, 0.2),
        ]
    )
    max_bf = np.max(
        [
            np.percentile(bf_center, 99.8),
            np.percentile(bf_top, 99.8),
            np.percentile(bf_lower_offset, 99.8),
            np.percentile(bf_upper_offset, 99.8),
        ]
    )

    # CDH5 (cdh5) min and max calculations
    min_cdh5 = np.min(
        [
            np.percentile(cdh5_center, 0.2),
            np.percentile(cdh5_top, 0.2),
            np.percentile(cdh5_lower_offset, 0.2),
            np.percentile(cdh5_upper_offset, 0.2),
        ]
    )
    max_cdh5 = np.max(
        [
            np.percentile(cdh5_center, 99.8),
            np.percentile(cdh5_top, 99.8),
            np.percentile(cdh5_lower_offset, 99.8),
            np.percentile(cdh5_upper_offset, 99.8),
        ]
    )

    # Contrast stretching
    bf_center = contrast_stretching(bf_center, custom_range=(min_bf, max_bf))
    bf_top = contrast_stretching(bf_top, custom_range=(min_bf, max_bf))
    bf_lower_offset = contrast_stretching(bf_lower_offset, custom_range=(min_bf, max_bf))
    bf_upper_offset = contrast_stretching(bf_upper_offset, custom_range=(min_bf, max_bf))

    cdh5_center = contrast_stretching(cdh5_center, custom_range=(min_cdh5, max_cdh5))
    cdh5_top = contrast_stretching(cdh5_top, custom_range=(min_cdh5, max_cdh5))
    cdh5_lower_offset = contrast_stretching(cdh5_lower_offset, custom_range=(min_cdh5, max_cdh5))
    cdh5_upper_offset = contrast_stretching(cdh5_upper_offset, custom_range=(min_cdh5, max_cdh5))

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

    plt.suptitle(f"{dataset_config.name} Position {POSITION} Timepoint {TIMEPOINT}\n")
    plt.tight_layout()
    plt.show()
    save_plot_to_path(fig, save_dir, f"{dataset_config.name}_pos{POSITION}_tp{TIMEPOINT}_im_slices")
    plt.close()

# %%
fig = plt.figure(figsize=(6, 6))
plt.hist(availabe_slices_above_list, bins=range(6, 20, 1), align="left", edgecolor="black")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Available Slices Above Center Slice")
plt.ylabel("Number of Datasets")
plt.show()
save_plot_to_path(fig, save_dir, "available_slices_above_center_histogram")
# %%
