# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from src.endo_pipeline.io import load_zarr_as_dask_array
from src.endo_pipeline.io.output import get_output_path, save_plot_to_path
from src.endo_pipeline.library.visualize.model_inputs.image_processing_steps import (
    process_brightfield,
)
from src.endo_pipeline.library.visualize.model_inputs.plot import visualize_images_with_histograms


# %%
def plot_mean_intensity_distribution(mean_list: list, save_dir: Path) -> None:
    """
    Plot the mean intensity distribution of brightfield images and saves the plot.

    Args:
        mean_list (list): List of mean intensity values.
        save_dir (path): Directory to save the plot.

    Returns:
        None. Saves the plot to the specified directory.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Round mean values and calculate unique values with counts
    rounded_mean_list = [round(x, -3) for x in mean_list]
    unique_values, counts = np.unique(rounded_mean_list, return_counts=True)

    # Convert unique values to string labels for the x-axis
    unique_labels = [str(int(val)) for val in unique_values]

    ax.bar(unique_labels, counts, color="gray", alpha=0.7)
    ax.set_title("Mean Intensity of Brightfield Images")
    ax.set_xlabel("Mean Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(unique_labels, rotation=45)

    fname = "mean_intensity_barplot_brightfield"
    save_plot_to_path(fig, save_dir, fname)


datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
mean_list = []
for dataset in datasets:
    config = load_dataset_config(dataset)
    position = 0
    timepoints = 0

    save_dir = get_output_path("model_input_visualization", "brightfield")
    zarr_file = get_zarr_file_for_position(config, position)
    bf_stack = load_zarr_as_dask_array(
        zarr_file, channels=["BF"], timepoints=timepoints, level=1, squeeze=True
    )
    bf_stack_float32 = bf_stack.astype("float32")
    bf_stack_float32_computed = bf_stack_float32.compute()

    mean_instensity = np.mean(bf_stack_float32_computed[15])
    mean_list.append(mean_instensity)

save_dir = get_output_path("bf_variation")
plot_mean_intensity_distribution(mean_list, save_dir)

#%%
dataset_list = ["20250730_3", "20250730_4", "20250730_5"]
for dataset in dataset_list:
    config = load_dataset_config(dataset)
    position = 0
    timepoints = 0

    save_dir = get_output_path("model_input_visualization", "brightfield")
    zarr_file = get_zarr_file_for_position(config, position)
    bf_stack = load_zarr_as_dask_array(
        zarr_file, channels=["BF"], timepoints=timepoints, level=1, squeeze=True
    )
    bf_stack_float32_computed, standard_dev_proj, clipped_im, normalized_im = process_brightfield(
        bf_stack
    )
    visualize_images_with_histograms(
        [
            ("BF Slice", bf_stack_float32_computed[15]),
            ("Std Dev Projection", standard_dev_proj),
            ("Clipped Std Dev Projection", clipped_im),
            ("Z-score Normalized Image", normalized_im),
        ],
        save_dir=save_dir,
        fname_prefix=f"{dataset}_P{position}_T{timepoints}_BF",
    )

# %% compare histograms
# low middle high
# Use Tableau colors from matplotlib
colors = plt.cm.tab10.colors  # Tableau 10 color palette

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
dataset_list = ["20250730_3", "20250730_4", "20250730_5"]

for i, dataset in enumerate(dataset_list):
    config = load_dataset_config(dataset)
    position = 0
    timepoints = 0

    save_dir = get_output_path("model_input_visualization", "brightfield")
    zarr_file = get_zarr_file_for_position(config, position)

    # STEP 1: Load the brightfield stack as a Dask array, convert to float32
    bf_stack = load_zarr_as_dask_array(
        zarr_file, channels=["BF"], timepoints=timepoints, level=1, squeeze=True
    )
    bf_stack_float32_computed, standard_dev_proj, clipped_im, normalized_im = process_brightfield(
        bf_stack
    )

    # Histogram for bf_stack_float32_computed
    data1 = bf_stack_float32_computed[14].ravel()
    mean1 = np.mean(data1)

    # Compute histogram and bin edges
    hist1, bin_edges1 = np.histogram(data1, bins=256)
    peak1_index = np.argmax(hist1)
    peak1 = (bin_edges1[peak1_index] + bin_edges1[peak1_index + 1]) / 2  # Midpoint of the peak bin

    # Plot filled histogram and matching outline
    axes[0].hist(data1, bins=256, alpha=0.25, label=f"Data: {dataset}", color=colors[i])
    axes[0].step(bin_edges1[:-1], hist1, where="mid", linewidth=2, label=f"Outline: {dataset}", color=colors[i])
    axes[0].axvline(mean1, color="blue", linestyle="--", label=f"Mean: {mean1:.2f}")
    axes[0].axvline(peak1, color="red", linestyle="--", label=f"Peak: {peak1:.2f}")
    axes[0].set_title("Histogram: BF Slice")
    axes[0].set_xlabel("Intensity Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(5000, 45000)
    axes[0].legend()

    # Histogram for normalized_im
    data2 = normalized_im.ravel()
    mean2 = np.mean(data2)

    # Compute histogram and bin edges
    hist2, bin_edges2 = np.histogram(data2, bins=256)
    peak2_index = np.argmax(hist2)
    peak2 = (bin_edges2[peak2_index] + bin_edges2[peak2_index + 1]) / 2  # Midpoint of the peak bin

    # Plot filled histogram and matching outline
    axes[1].hist(data2, bins=256, alpha=0.25, label=f"Data: {dataset}", color=colors[i])
    axes[1].step(bin_edges2[:-1], hist2, where="mid", linewidth=2, label=f"Outline: {dataset}", color=colors[i])
    axes[1].axvline(mean2, color="blue", linestyle="--", label=f"Mean: {mean2:.2f}")
    axes[1].axvline(peak2, color="red", linestyle="--", label=f"Peak: {peak2:.2f}")
    axes[1].set_title("Histogram: Z-score Normalized Standard Deviation BF Projection")
    axes[1].set_xlabel("Intensity Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(-2, 8)
    axes[1].legend()

plt.tight_layout()
plt.show()

fname = "sanity_check_z_score_normalized_bf_histograms"
save_plot_to_path(fig, save_dir, fname)
# %%
