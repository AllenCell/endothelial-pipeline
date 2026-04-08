from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """Validate dataset(s) by checking config schemas and loading files.

    #test-ready #cpu-only

    For each specified dataset, confirm:

    - The dataset config exists and can be loaded
    - All dataset configs follow the schema defined by `DatasetConfig`
    - All original data paths exist and can be opened
    - All zarr data paths exist and can be opened
    - All shear stress regimes are valid based on the flow conditions

    If `datasets` is not provided, all datasets will be validated.

    Parameters
    ----------
    datasets
        The dataset(s) to validate. If None, all datasets will be validated.

    """
    import logging
    from pathlib import Path

    from bioio import BioImage

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_available_dataset_names,
        load_dataset_config,
        validate_dataset_config,
    )
    from endo_pipeline.manifests import get_zarr_location_for_position

    logger = logging.getLogger(__name__)

    dataset_names = datasets or get_available_dataset_names()
    if DEMO_MODE:
        # Each dataset takes 2-9 seconds to validate
        dataset_names = dataset_names[:2]

    for dataset_name in dataset_names:
        logger.info(f"Running validation for dataset [ {dataset_name} ]")

        # Validate dataset config schema.
        validate_dataset_config(dataset_name)

        # Load dataset config.
        dataset_config = load_dataset_config(dataset_name)

        # Check if file at original path exists and can be opened.
        try:
            BioImage(Path(dataset_config.original_path))
        except FileNotFoundError:
            logger.error(
                "Failed to open original for dataset [ %s ] at [ %s ]",
                dataset_config.name,
                dataset_config.original_path,
            )

        # For each position, check if the local zarr exists and can be opened.
        for position in dataset_config.zarr_positions:
            zarr_file = get_zarr_location_for_position(dataset_config, position).path

            try:
                BioImage(zarr_file)
            except:
                logger.error(
                    "Failed to load zarr for dataset [ %s ] at [ %s ]", dataset_name, zarr_file
                )
                raise


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
