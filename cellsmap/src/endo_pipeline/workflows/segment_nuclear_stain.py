import argparse
import os

import matplotlib.pyplot as plt
import tifffile
from cellpose import models
from skimage.color import label2rgb

from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import dataset_io, set_output
from cellsmap.vis import get_images, image_processing

"""
Segment nuclear stain channel using Cellpose for immunofluorescence datasets.

To test this script, you can run it with the following command:
python cellsmap/src/endo_pipeline/workflows/segment_nuclear_stain.py \
    --dataset "20250509_20X_IF2" \
    --nuc_stain "NucViolet"

To run this script on new datasets, you can use the following command:
python cellsmap/src/endo_pipeline/workflows/segment_nuclear_stain.py \
    --dataset "20250509_20X_IF2" \
    --nuc_stain "NucViolet" \
    --output_dir "//allen/aics/endothelial/morphological_features/segmentations/nuclear_stain_seg/"

"""


# %%
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
    Visualize the original images, segmentation masks, and overlays.

    Args:
        max_int_projections (list): List of maximum intensity projections.
        masks (list): List of segmentation masks.
        dataset (str): Dataset name.
    """
    for max_int_proj, mask in zip(max_int_projections, masks, strict=True):

        colored_mask = label2rgb(mask, bg_label=0, kind="overlay")

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(max_int_proj, cmap="gray")
        axes[0].set_title("Original Flourescence Image", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(colored_mask)
        axes[1].set_title("Nuclear Segmentation Mask", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(max_int_proj, cmap="gray")  # Show original DAPI first
        axes[2].imshow(colored_mask, alpha=0.2)  # Add the colored mask with transparency overlay
        axes[2].set_title("Overlay", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        output_path = set_output.get_output_path("nuclear_stain_segmentation")
        vb.save_plot(fig, output_path + f"{dataset}_segmentation_overlay")


def save_segmentation_masks(masks: list, dataset: str, output_dir: str) -> None:
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
        save_path = f"{output_dir}/{dataset}/P{position}/"
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        tifffile.imwrite(save_path + f"{dataset}_P{position}_T0.ome.tiff", mask)


def process_dataset(
    dataset: str, nuc_stain: str, output_dir: str | None = None, visualize: bool = True
) -> None:
    """
    Full workflow to process a dataset: projection, segmentation, visualization, and saving.

    Args:
        dataset (str): Dataset name.
        nuc_stain (str): Nuclear stain channel name.
        output_dir (str): Directory to save the results. If None, uses default output directory.
        visualize (bool): Whether to plot the results.
    """
    print(f"Processing {dataset}...")

    # Step 1: Get maximum intensity projections
    max_int_projections = get_max_int_projections(dataset, nuc_stain)

    # Step 2: Perform nuclear segmentation
    masks = segment_nuclei(max_int_projections)

    # Step 3: Visualize results (optional)
    if visualize:
        visualize_results(max_int_projections, masks, dataset)

    # Step 4: Save segmentation masks
    if output_dir is None:
        output_dir = set_output.get_output_path("nuclear_stain_segmentation")
    else:
        print(f"Outputs saved to {output_dir}")

    save_segmentation_masks(masks, dataset, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process nuclear stain dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to process (e.g., '20250509_20X_IF2').",
    )
    parser.add_argument(
        "--nuc_stain",
        type=str,
        required=True,
        help="Name of the nuclear stain (e.g., 'NucViolet', 'DAPI').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the results. If not provided, uses default output directory.",
    )

    args = parser.parse_args()

    process_dataset(args.dataset, args.nuc_stain, args.output_dir, visualize=True)
