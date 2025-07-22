from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.endo_pipeline.io import save_plot_to_path
from src.endo_pipeline.library.process.image_processing import contrast_stretching


def plot_standard_devs_per_slice(
    stdevs: list, center_plane: int, dataset: str, position: int, frame: int, output_dir: Path
) -> None:
    """
    Plot the standard deviations of each slice vs plane index, highlighting the center plane.

    Args:
        stdevs (list): A list of standard deviation values for each bf plane in the z-stack.
        center_plane (int): The index of the center plane to highlight on the plot.
        dataset (str): The name of the dataset.
        position (int): The position index.
        frame (int): The frame index.
        output_dir (Path): The directory where the plot will be saved.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stdevs)

    # Add a vertical line at the center plane index
    ax.axvline(center_plane, color="red", linestyle="--", label=f"Center Plane ({center_plane})")

    ax.set_title(f"{dataset} P{position} frame: {frame}")
    ax.set_xlabel("Plane Index")
    ax.set_ylabel("Standard Deviations of BF Planes")
    ax.legend()  # Add legend for the center plane label

    fname = f"standard_devs_{dataset}_P{position}_{frame}"
    save_plot_to_path(fig, output_dir, fname)
    plt.show()


def visualize_slice_selection(
    bf_stack: np.ndarray,
    cdh5_stack: np.ndarray,
    center_plane: int,
    lower_offset: int,
    upper_offest: int,
    dataset: str,
    position: int,
    frame: int,
    output_dir: Path,
) -> None:
    """
    Plot the center z-slice with slices n planes above (upper offset) and below (lower offset)
    the center slice for the bf and chd5 channels.

    Args:
        bf_stack (np.ndarray): Brightfield image stack.
        cdh5_stack (np.ndarray): CDH5 image stack.
        center_plane (int): Index of the center plane.
        lower_offset (int): Number of planes below the center plane to visualize.
        upper_offest (int): Number of planes above the center plane to visualize.
        dataset (str): Dataset name.
        frame (int): Frame index.
        position (int): Zarr Position index.
        output_dir (path): Directory to save the output plot.

    Returns:
        None
    """
    im_center = contrast_stretching(bf_stack[center_plane].compute())
    im_below = contrast_stretching(bf_stack[center_plane - lower_offset].compute())
    im_above = contrast_stretching(bf_stack[center_plane + upper_offest].compute())

    cdh5_center = contrast_stretching(cdh5_stack[center_plane].compute())
    cdh5_below = contrast_stretching(cdh5_stack[center_plane - lower_offset].compute())
    cdh5_above = contrast_stretching(cdh5_stack[center_plane + upper_offest].compute())

    # Create subplots with a 2x3 grid
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 10)
    )  # Adjusted figure size for 2 rows and 3 columns

    # First row: BF stack
    axes[0, 0].imshow(im_below, cmap="gray")
    axes[0, 0].set_title(f"BF Plane {center_plane - lower_offset} (-{lower_offset})")

    axes[0, 1].imshow(im_center, cmap="gray")
    axes[0, 1].set_title(f"BF Center Plane {center_plane}")

    axes[0, 2].imshow(im_above, cmap="gray")
    axes[0, 2].set_title(f"BF Plane {center_plane + upper_offest} (+{upper_offest})")

    # Second row: CDH5 stack
    axes[1, 0].imshow(cdh5_below, cmap="gray")
    axes[1, 0].set_title(f"CDH5 Plane {center_plane - lower_offset} (-{lower_offset})")

    axes[1, 1].imshow(cdh5_center, cmap="gray")
    axes[1, 1].set_title(f"CDH5 Center Plane {center_plane}")

    axes[1, 2].imshow(cdh5_above, cmap="gray")
    axes[1, 2].set_title(f"CDH5 Plane {center_plane + upper_offest} (+{upper_offest})")

    for ax in axes.flat:
        ax.axis("off")

    # Adjust layout and add a title
    fig.suptitle(f"{dataset} P{position} Frame {frame}", fontsize=16)
    plt.tight_layout()

    # Save the plot
    fname = f"plane_selection_vis_{dataset}_P{position}_{frame}_offset{lower_offset}_{upper_offest}"
    save_plot_to_path(fig, output_dir, fname)
    plt.show()


def plot_global_center_plane(
    center_planes: list, dataset: str, position: int, output_dir: Path
) -> tuple[float, float]:
    """
    Plot center planes for a dataset and return mean center plane and standard deviation.

    Args:
        center_planes (list): List of center plane indices.
        dataset (str): Dataset name for labeling the plot.
        position (int): Position index for labeling the plot.
        output_dir (Path): Directory to save the output plot.

    Returns:
        tuple (float, float): Mean and standard deviation of center planes.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))  # Create two subplots

    # Compute mean and standard deviation of center planes
    mean_center_plane = np.mean(center_planes)
    std_center_plane = np.std(center_planes)

    # First plot: Center plane vs frame index
    ax[0].plot(range(len(center_planes)), center_planes, "ro", alpha=0.7)
    ax[0].set_xlabel("Timepoint (frames)")
    ax[0].set_ylabel("Center Plane Index")
    ax[0].set_title(f"{dataset}, Position {position}")
    ax[0].set_ylim(0, 25)

    # Add horizontal lines for mean and ±1 standard deviation
    ax[0].axhline(
        mean_center_plane, color="black", linestyle="-", label=f"Mean: {mean_center_plane:.2f}"
    )
    ax[0].axhline(
        mean_center_plane - std_center_plane,
        color="cyan",
        linestyle="--",
        label=f"-1 Std Dev: {mean_center_plane - std_center_plane:.2f}",
    )
    ax[0].axhline(
        mean_center_plane + std_center_plane,
        color="cyan",
        linestyle="--",
        label=f"+1 Std Dev: {mean_center_plane + std_center_plane:.2f}",
    )
    ax[0].legend()

    # Second plot: Histogram of center planes
    ax[1].hist(center_planes, bins=25, color="gray", alpha=0.7)
    ax[1].axvline(
        mean_center_plane, color="black", linestyle="-", label=f"Mean: {mean_center_plane:.2f}"
    )
    ax[1].axvline(mean_center_plane - std_center_plane, color="cyan", linestyle="--")
    ax[1].axvline(mean_center_plane + std_center_plane, color="cyan", linestyle="--")
    ax[1].set_xlabel("Center Plane Index")
    ax[1].set_xlim(0, 25)
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    fname = f"global_center_plane_{dataset}_P{position}"
    save_plot_to_path(fig, output_dir, fname)
    plt.show()

    return mean_center_plane, std_center_plane
