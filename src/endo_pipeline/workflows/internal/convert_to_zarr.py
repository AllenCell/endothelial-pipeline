from endo_pipeline.cli import Datasets, tags

TAGS = ["internal", tags.TEST_READY, tags.CPU_ONLY]


def main(
    datasets: Datasets | None = None,
    output_path: str | None = None,
    channel_names: list[str] = ["EGFP", "BF"],
) -> None:
    """
    Convert datasets to Zarr format.

    Parameters
    ----------
    datasets
        List of dataset names to convert. If None, defaults to a predefined collection.
    output_path
        Path to save the converted Zarr files. If None, defaults to a standard output path.
        Save zarrs to //allen/aics/endothelial/morphological_features/image_data/converted_zarrs
        for pipeline use.
    channel_names
        List of channel names to include in the Zarr files.
        ie ["EGFP", "BF", "NucViolet", "SOX17", "SMAD1"] for SMAD1 IF data.
        Zarrs are saved in this default channel order: 488, BF, 405, 561, 640.
    """
    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.convert_to_zarr.convert_dataset import convert_dataset

    logger = logging.getLogger(__name__)

    if datasets is None:
        logger.info("No datasets provided. Using first dataset in default collection.")
        datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
        datasets = datasets[:1]

    if DEMO_MODE or output_path is None:
        logger.info("DEMO_MODE is ON or no output path provided. Using default output path.")
        output_path = get_output_path("zarr_conversion")

    for dataset_name in datasets:
        logger.info(f"Converting dataset: {dataset_name}")

        convert_dataset(
            dataset=dataset_name,
            output_dataset_name=dataset_name[:8],
            output_path=output_path,
            channel_names=channel_names,
            demo_mode=DEMO_MODE,
        )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
