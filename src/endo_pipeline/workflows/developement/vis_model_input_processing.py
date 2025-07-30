#%%
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
from src.endo_pipeline.library.process.image_processing import (
    clip_image,
    max_proj,
    scale_intensity_range_percentiles,
    std_dev,
    z_score_normalize_intensity,
)


#%%
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

    ax.bar(unique_labels, counts, color='gray', alpha=0.7)
    ax.set_title("Mean Intensity of Brightfield Images")
    ax.set_xlabel("Mean Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(unique_labels, rotation=45)

    fname="mean_intensity_barplot_brightfield"
    save_plot_to_path(fig, save_dir, fname)


def visualize_images_with_histograms(
    images: list[tuple[str, np.ndarray]],
    save_dir: Path,
    fname_prefix: str
) -> None:
    """
    Visualize multiple images alongside their histograms in a grid and save the plot.

    Args:
        images (List[Tuple[str, np.ndarray]]): List of (title, image) tuples.
        save_dir (Path): Directory where the plot will be saved.
        fname_prefix (str): Filename prefix for the saved figure.

    Returns:
        None
    """
    rows = len(images)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 16))

    for row_idx, (title, image) in enumerate(images):
        # Left: Image
        axes[row_idx, 0].imshow(image, cmap='gray')
        axes[row_idx, 0].set_title(title)
        axes[row_idx, 0].axis('off')

        # Right: Histogram
        hist, bin_edges = np.histogram(image.ravel(), bins=256)
        mean_intensity = np.mean(image)
        peak_intensity = bin_edges[np.argmax(hist)]

        axes[row_idx, 1].hist(image.ravel(), bins=256, color='gray')
        axes[row_idx, 1].set_title(f"Histogram of {title}")
        axes[row_idx, 1].set_xlabel("Intensity")
        axes[row_idx, 1].set_ylabel("Pixel count")

        # Annotate mean intensity
        axes[row_idx, 1].axvline(mean_intensity, color='blue', linestyle='--', label=f"Mean: {mean_intensity:.2f}")
        # Annotate peak intensity
        axes[row_idx, 1].axvline(peak_intensity, color='red', linestyle='--', label=f"Peak: {peak_intensity:.2f}")

        # Add legend
        axes[row_idx, 1].legend()

    plt.suptitle(f"{fname_prefix}", fontsize=16)
    plt.tight_layout()
    plt.show()

    fname = f"{fname_prefix}_image_process_histogram"
    save_plot_to_path(fig, save_dir, fname)

#%%
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
mean_list = []
for dataset in datasets:
    config = load_dataset_config(dataset)
    position = 0
    timepoints = 0
    
    save_dir = get_output_path("model_input_visualization", "brightfield")
    zarr_file = get_zarr_file_for_position(config, position)

    # STEP 1: Load the brightfield stack as a Dask array, convert to float32
    bf_stack = load_zarr_as_dask_array(
        zarr_file,
        channels=["BF"], 
        timepoints=timepoints, 
        level=1, 
        squeeze=True
    )
    bf_stack_float32 = bf_stack.astype("float32")
    bf_stack_float32_computed = bf_stack_float32.compute()

    # STEP 2: Standard deviation projection along the Z-axis
    standard_dev_proj = std_dev(bf_stack_float32, axis=0, unbiased=False).astype("float32")

    # STEP 3: Clip image by percentiles
    clipped_im = clip_image(standard_dev_proj, low_pct=0.1, high_pct=99.9)

    # STEP 4: Z-score normalize
    normalized_im = z_score_normalize_intensity(clipped_im)

    # Visualize steps
    image_processing_steps = [
        ("BF Slice", bf_stack_float32_computed[15]),
        ("Std Dev Projection", standard_dev_proj),
        ("Clipped Std Dev Projection", clipped_im),
        ("Z-score Normalized Image", normalized_im),
    ]
    visualize_images_with_histograms(
        image_processing_steps,
        save_dir=save_dir,
        fname_prefix=f"{dataset}_P{position}_T{timepoints}_BF"
    )
    
    # calculate the peak intensity of the histogram bf_stack_float32_computed[15]
    mean_instensity = np.mean(bf_stack_float32_computed[15])
    mean_list.append(mean_instensity)


#%%
save_dir = get_output_path("bf_variation")
plot_mean_intensity_distribution(mean_list, save_dir)


# %% CDH5 Visualization
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
for dataset in datasets:
    config = load_dataset_config(dataset)
    position = 0
    timepoints = 0
    
    save_dir = get_output_path("model_input_visualization", "cdh5")
    zarr_file = get_zarr_file_for_position(config, position)

    # STEP 1: Load the brightfield stack as a Dask array, convert to float32
    cdh5_stack = load_zarr_as_dask_array(
        zarr_file,
        channels=["EGFP"], 
        timepoints=timepoints, 
        level=1, 
        squeeze=True
    )
    cdh5_stack_float32 = cdh5_stack.astype("float32")
    cdh5_stack_float32_computed = cdh5_stack_float32.compute()

    # STEP 2: Maximum projection along the Z-axis
    max_proj_im = max_proj(cdh5_stack_float32, axis=0).astype("float32")
    
    # STEP 3: Clip image by percentiles, Map linearly -1 to 1
    scaled_im = scale_intensity_range_percentiles(max_proj_im, 10, 98, -1.0, 1.0, clip=True)
    
    image_processing_steps = [
        ("CDH5 Slice", cdh5_stack_float32_computed[15]),
        ("Max Projection", max_proj_im),
        ("Scaled, Clipped Image", scaled_im),
    ]
    visualize_images_with_histograms(
        image_processing_steps,
        save_dir=save_dir,
        fname_prefix=f"{dataset}_P{position}_T{timepoints}_CDH5"
    )
# %% Load images 
