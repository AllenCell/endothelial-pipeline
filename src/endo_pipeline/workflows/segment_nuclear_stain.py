import argparse
from pathlib import Path

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.library.process.if_segmentation import (
    get_max_int_projections,
    save_segmentation_masks,
    segment_nuclei,
    visualize_results,
)

"""
Segment nuclear stain channel using Cellpose for immunofluorescence datasets.

To test this script, you can run it with the following command (~5 min):
python src/endo_pipeline/workflows/segment_nuclear_stain.py \
    --dataset "20250522_20X_IFA" \
    --nuc_stain "NucViolet"

To run this script on new datasets, you can use the following command:
python src/endo_pipeline/workflows/segment_nuclear_stain.py \
    --dataset "20250522_20X_IFA" \
    --nuc_stain "NucViolet" \
    --output_dir "//allen/aics/endothelial/morphological_features/segmentations/nuclear_stain_seg/"

"""


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
    max_int_projections, xy_pixel_size_um = get_max_int_projections(dataset, nuc_stain)

    # Step 2: Perform nuclear segmentation
    masks = segment_nuclei(max_int_projections)

    # Step 3: Visualize results (optional)
    if visualize:
        visualize_results(max_int_projections, masks, dataset)

    # Step 4: Save segmentation masks
    if output_dir is None:
        output_path = get_output_path("nuclear_stain_segmentation")
    else:
        output_path = Path(output_dir)
        print(f"Outputs saved to {output_dir}")

    datset_config = load_dataset_config(dataset)
    save_segmentation_masks(masks, datset_config, output_path, xy_pixel_size_um)


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
