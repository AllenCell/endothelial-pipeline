from endo_pipeline.cli import Datasets

TAGS = ["immunofluorescence"]


def main(datasets: Datasets | None = None) -> None:
    """
    Generate an immunofluorescence manifest for a given dataset.

    Use demo mode to process one position and skip uploading to FMS / updating manifest config.

    Parameters
    ----------
    datasets
        List of dataset names to process. If None, processes all datasets in the
        "live_20X_objective_3i_microscope" collection. If DEMO_MODE is enabled,
        only the first dataset will be processed.
    """
    import logging

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.library.process.if_feature_extraction import run_nuclei_feature_extraction
    from endo_pipeline.library.process.if_manifest import (
        save_manifest_to_csv,
        update_dataframe_manifest,
        upload_manifest_to_fms,
    )

    logger = logging.getLogger(__name__)

    if datasets is None:
        datasets = get_datasets_in_collection("immunofluorescence")

    if DEMO_MODE:
        logger.info("DEMO_MODE enabled: only processing the first dataset")
        datasets = datasets[:1]

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")

        # Step 1: Run feature extraction
        df = run_nuclei_feature_extraction(dataset)

        # Step 2: Save to CSV
        save_path = save_manifest_to_csv(dataset, df)

        if DEMO_MODE:
            logger.info("DEMO_MODE mode enabled. Skipping fms upload and dataset config update")
            continue

        # Step 3: Upload to FMS
        fms_id = upload_manifest_to_fms(save_path, dataset)

        # Step 4: Update dataset configuration
        update_dataframe_manifest(dataset, fms_id)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
