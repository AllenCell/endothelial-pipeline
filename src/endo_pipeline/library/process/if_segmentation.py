import os
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile
from cellpose import models
from skimage.color import label2rgb

from endo_pipeline.configs import dataset_io
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.process import get_images, image_processing


def get_max_int_projections(dataset: str, nuc_stain: str) -> list:
    """
    Get maximum intensity projections of nuclear stain channel for all positions in the dataset.

    Args:
        dataset (str): Dataset name.
        nuc_stain (str): Nuclear stain channel name. (ie. "NucViolet", "DAPI")

    Returns:
        list: List of maximum intensity projections.
    """
    n_positions = dataset_io.get_total_number_of_positions(dataset)

    max_int_projections = []
    for position in range(n_positions):
        print(f"Processing position {position}...")

        img = get_images.get_zarr_img_for_dataset(dataset, position, resolution_level=0)
        channel_names = img.channel_names
        nuc_stain_channel = channel_names.index(nuc_stain)

        img_tp = img.get_image_dask_data("ZYX", T=0, C=nuc_stain_channel)
        max_int_projection = image_processing.max_proj(img_tp, axis=0)
        max_int_projections.append(max_int_projection)

    return max_int_projections


def segment_nuclei(max_int_projections: list) -> list:
    """
    Perform nuclear segmentation using Cellpose on nuclear stain maximum intensity projections.

    Args:
        max_int_projections (list): List of maximum intensity projections.

    Returns:
        masks (list): List of segmentation masks for each maximum intensity projection.
    """
    print("Segmenting nuclei...")
    model = models.Cellpose(model_type="nuclei")
    masks, _flows, _styles, _diams = model.eval(max_int_projections, diameter=None, channels=[0, 0])
    return masks


def visualize_results(max_int_projections: list, masks: list, dataset: str) -> None:
    """
    Visualize the contrast stretched images, segmentation masks, and overlays.

    Args:
        max_int_projections (list): List of maximum intensity projections.
        masks (list): List of segmentation masks.
        dataset (str): Dataset name.
    """
    for max_int_proj, mask in zip(max_int_projections, masks, strict=True):

        colored_mask = label2rgb(mask, bg_label=0, kind="overlay")

        contrasted_max_int_proj = image_processing.contrast_stretching(max_int_proj, "percentile")

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(contrasted_max_int_proj, cmap="gray")
        axes[0].set_title("Flourescence Image", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(colored_mask)
        axes[1].set_title("Nuclear Segmentation Mask", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(contrasted_max_int_proj, cmap="gray")  # Show contrasted signal
        axes[2].imshow(colored_mask, alpha=0.2)  # Add the colored mask with transparency overlay
        axes[2].set_title("Overlay", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        output_path = get_output_path("nuclear_stain_segmentation")
        save_plot_to_path(fig, output_path, f"{dataset}_segmentation_overlay")


def save_segmentation_masks(masks: list, dataset: str, output_dir: Path) -> None:
    """
    Save segmentation masks as TIFF files.

    Args:
        masks (list): List of segmentation masks.
        dataset (str): Dataset name.
        output_dir (str): Directory to save the masks.
    """
    print("Saving segmentation masks...")
    n_positions = dataset_io.get_total_number_of_positions(dataset)
    for mask, position in zip(masks, range(n_positions), strict=True):
        save_path = output_dir / dataset / f"P{position}"
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        tifffile.imwrite(save_path / f"{dataset}_P{position}_T0.ome.tiff", mask)
