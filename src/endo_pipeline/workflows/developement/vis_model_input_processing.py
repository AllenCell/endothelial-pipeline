#%%
import matplotlib.pyplot as plt
import numpy as np

from src.endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from src.endo_pipeline.io import load_zarr_as_dask_array
from src.endo_pipeline.library.process.image_processing import (
    clip_image,
    std_dev,
    z_score_normalize_intensity,
)

from src.endo_pipeline.io.output import save_plot_to_path, get_output_path


#%%
def visualize_image_and_histogram(image: np.ndarray, title: str) -> None:
    """
    Visualize an image alongside its intensity histogram.

    Args:
        image (np.ndarray): The input image to be visualized. It should be a 2D array (grayscale image).
        title (str): The title for the image and histogram.

    Returns:
        None: This function displays the image and histogram but does not return any value.
    """
    plt.figure(figsize=(12, 5))

    # Image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(image.ravel(), bins=256, color='gray')
    plt.title(f"Histogram of {title}")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")

    plt.tight_layout()
    plt.show()

#%%
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

for dataset in datasets:
    config = load_dataset_config(dataset)
    position = 0
    timepoints = 500
    zarr_file = get_zarr_file_for_position(config, position)

    # STEP 0: Load the brightfield stack as a Dask array
    bf_stack = load_zarr_as_dask_array(
        zarr_file,
        channels=["BF"], 
        timepoints=timepoints, 
        level=1, 
        squeeze=True
    )

    # STEP 1: Convert to float32
    bf_stack_float32 = bf_stack.astype("float32")
    bf_stack_float32_computed = bf_stack_float32.compute()
    visualize_image_and_histogram(bf_stack_float32_computed[15], 
                                  f"BF Stack (float32), frame {timepoints}")

    # STEP 2: Standard deviation projection along the Z-axis
    standard_dev_proj = std_dev(bf_stack_float32, axis=0, unbiased=False).astype("float32")
    visualize_image_and_histogram(standard_dev_proj, "Std Dev Projection")

    # STEP 3: Clip image by percentiles
    clipped_im = clip_image(standard_dev_proj, low_pct=0.1, high_pct=99.9)
    visualize_image_and_histogram(clipped_im, "Clipped Std Dev Projection")

    # STEP 4: Z-score normalize
    normalized_im = z_score_normalize_intensity(clipped_im)
    visualize_image_and_histogram(normalized_im, "Z-score Normalized Image")
    break

# %%
