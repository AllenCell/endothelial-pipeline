from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.endo_pipeline.io.output import save_plot_to_path


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