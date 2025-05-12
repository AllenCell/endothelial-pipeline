from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb

from cellsmap.analyses.utils.viz import viz_base as vb


# %%
def projection_image(
    sum_proj_img: np.ndarray,
    seg_mask: np.ndarray,
    dataset: str,
    position: str,
    start_x: int,
    start_y: int,
    output_dir: str,
) -> None:
    """
    Display the sum projection, segmentation mask, and sum projection in nuclei
    in a single figure with subplots.

    Parameters:
        sum_proj_img (np.ndarray): The sum projection image.
        seg_mask (np.ndarray): The segmentation mask.
        sum_proj_img_in_nuclei (np.ndarray): The sum projection image within nuclei.
    """

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    colored_mask = label2rgb(seg_mask, bg_label=0, kind="overlay")

    axes[0].imshow(sum_proj_img, cmap="gray")
    axes[0].set_title("Sum Projection")
    axes[0].axis("off")

    axes[1].imshow(colored_mask)
    axes[1].set_title("Nuclear Segmentation Mask")
    axes[1].axis("off")

    axes[2].imshow(sum_proj_img, cmap="gray")
    axes[2].imshow(colored_mask, alpha=0.2)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    fname = f"{dataset}_{position}_crop_{start_x}_{start_y}_projection"
    vb.save_plot(fig, output_dir + fname)


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

        for j in range(num_slices, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()


def plot_intensity_distribution(
    df: pd.DataFrame,
    xlabel: str,
    dataset: str,
    output_dir: str,
    xlim: Optional[int] = None,
    ylim: Optional[int] = None,
) -> None:
    """
    Plot the distribution of a specified intensity feature from the DataFrame.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(df[xlabel].to_numpy(), bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("Frequency")
    plt.show()
    fname = f"{dataset}_{xlabel}_distribution"
    vb.save_plot(fig, output_dir + fname)
