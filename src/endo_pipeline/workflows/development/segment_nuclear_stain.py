from pathlib import Path

from endo_pipeline.cli import Datasets, tags

TAGS = ["immunofluorescence", tags.TEST_READY, tags.CPU_ONLY]


def main(
    datasets: Datasets | None = None,
    nuc_stain: str = "NucViolet",
    output_dir: Path | None = None,
    visualize: bool = True,
) -> None:
    """
    Segment nuclear stain channel using Cellpose for immunofluorescence datasets.

    To save segmentation masks to program folder use:
    --output_dir //allen/aics/endothelial/morphological_features/segmentations/nuclear_stain_seg/multiscale_zarr/

    Args:
        dataset (str): Dataset name.
        nuc_stain (str): Nuclear stain channel name (ie "NucViolet", "DAPI").
        output_dir (str): Directory to save the results. If None, uses default output directory.
        visualize (bool): Whether to plot the results.
    """
    import logging

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.if_segmentation import (
        get_max_int_projections,
        save_segmentation_masks,
        segment_nuclei,
        visualize_results,
    )
    from endo_pipeline.manifests import ImageLocation, load_image_manifest, save_image_manifest

    logger = logging.getLogger(__name__)

    if datasets is None:
        datasets = get_datasets_in_collection("immunofluorescence")

    if DEMO_MODE:
        logger.info("DEMO MODE enabled: only processing the first dataset")
        datasets = datasets[:1]

    for dataset_name in datasets:
        logging.info(f"Processing dataset: {dataset_name}")
        dataset_config = load_dataset_config(dataset_name)

        positions = dataset_config.zarr_positions
        if DEMO_MODE:
            positions = positions[:1]

        # Step 1: Get maximum intensity projections
        max_int_projections, xy_pixel_size_um = get_max_int_projections(
            dataset_config, nuc_stain, positions
        )

        # Step 2: Perform nuclear segmentation
        masks = segment_nuclei(max_int_projections)

        # Step 3: Visualize results (optional)
        if visualize:
            visualize_results(max_int_projections, masks, dataset_name)

        # # Step 4: Save segmentation masks
        if DEMO_MODE or output_dir is None:
            output_path = get_output_path("nuclear_stain_segmentation")
        else:
            output_path = Path(output_dir)
            logger.info(f"Outputs saved to {output_path}")

        save_segmentation_masks(masks, dataset_config, output_path, xy_pixel_size_um, positions)

        # Step 5: Update image manifest
        if not DEMO_MODE:
            img_manifest = load_image_manifest("nuclear_stain_seg")

            date = dataset_config.name[:8]
            fmsid = dataset_config.fmsid
            suffix = "P{{position}}.ome.zarr"
            new_path = f"{output_path}/{date}_{fmsid}/{date}_{fmsid}_{suffix}"
            img_manifest.locations[dataset_config.name] = ImageLocation(path=Path(new_path))

            save_image_manifest(img_manifest)


if __name__ == "__main__":

    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
