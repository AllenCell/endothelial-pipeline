import numpy as np
import matplotlib.pyplot as plt
from typing import List
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd


# %%
def projection_image(
    sum_proj_img: np.ndarray,
    seg_mask: np.ndarray,
    sum_proj_img_in_nuclei: np.ndarray,
    sum_projection_not_in_nuclei: np.ndarray,
) -> None:
    """
    Display the sum projection, segmentation mask, and sum projection in nuclei
    in a single figure with subplots.

    Parameters:
        sum_proj_img (np.ndarray): The sum projection image.
        seg_mask (np.ndarray): The segmentation mask.
        sum_proj_img_in_nuclei (np.ndarray): The sum projection image within nuclei.
    """
    images: List[np.ndarray] = [
        sum_proj_img,
        seg_mask,
        sum_proj_img_in_nuclei,
        sum_projection_not_in_nuclei,
    ]
    titles: List[str] = [
        "Sum Projection",
        "Segmentation Mask",
        "Sum Projection in Nuclei",
        "Sum Projection Cytoplasmic",
    ]

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


# %%
def histogram_intensity_per_slice(
    raw_crop: np.ndarray,
    background_subtracted_crop: np.ndarray,
) -> None:
    """
    Plot histograms of pixel intensities for each slice in the image.

    Parameters:
        raw_crop (np.ndarray): The raw cropped image stack (3D array).
        background_subtracted_crop (np.ndarray): The background-subtracted cropped image stack (3D array).
    """
    # Create a figure with subplots
    plt.figure(figsize=(15, 10))

    # Loop through each slice and plot its histogram
    for img in [raw_crop, background_subtracted_crop]:
        num_slices: int = img.shape[0]

        # Calculate the number of rows and columns for the subplots
        num_cols: int = (
            num_slices + 1
        ) // 2  # Two rows, so divide slices by 2 and round up
        num_rows: int = 2

        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10), sharey=True)

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Loop through each slice and plot its histogram
        for i in range(num_slices):
            hist, bin_edges = np.histogram(img[i, :, :].flatten().compute(), bins=100)

            # Plot histogram on the corresponding subplot
            axes[i].bar(
                bin_edges[:-1],
                hist,
                width=np.diff(bin_edges),
                edgecolor="black",
                align="edge",
            )
            axes[i].set_title(f"Slice {i}")
            axes[i].set_xlabel("Pixel Intensity")
            axes[i].set_ylabel(
                "Frequency" if i % num_cols == 0 else ""
            )  # Only show y-label on the first column

        # Hide any unused subplots
        for j in range(num_slices, len(axes)):
            axes[j].axis("off")

        # Adjust layout
        plt.tight_layout()
        plt.show()


def plot_intensity_distribution(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str = "Frequency",
) -> None:
    
    plt.figure(figsize=(10, 10))
    plt.hist(df[xlabel].to_numpy(), bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
