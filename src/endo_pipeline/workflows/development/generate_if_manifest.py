from endo_pipeline.cli import Datasets, tags

TAGS = ["immunofluorescence", tags.TEST_READY, tags.CPU_ONLY]


def main(datasets: Datasets | None = None) -> None:
    """
    Generate an immunofluorescence manifest for a given dataset.

    Use demo mode to process one position and skip uploading to FMS / updating manifest config.

    Parameters
    ----------
    datasets
        List of dataset names to process. If None, processes all datasets in the
        "immunofluorescence" collection. If DEMO_MODE is enabled,
        only the first position in the first dataset will be processed.
    """
    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
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
        dataset_config = load_dataset_config(dataset)

        positions = dataset_config.zarr_positions
        if DEMO_MODE:
            positions = positions[:1]

        # Step 1: Run feature extraction
        df = run_nuclei_feature_extraction(dataset_config, positions)

        # add data info to dataframe
        df["date"] = dataset_config.date
        shear_regime = "_to_".join([shear.value for shear in dataset_config.shear_stress_regime])
        df["shear_stress_regime"] = shear_regime

        shear_stress_list = [condition.shear_stress for condition in dataset_config.flow_conditions]
        df["shear_stress_1"] = shear_stress_list[0]
        df["shear_stress_2"] = shear_stress_list[1] if len(shear_stress_list) > 1 else np.nan

        durations = [
            condition.stop - condition.start for condition in dataset_config.flow_conditions
        ]
        duration_1 = durations[0]
        duration_2 = durations[1] if len(durations) > 1 else np.nan

        df["duration_at_ss_1_hr"] = duration_1 * 5 / 60  # convert to hrs
        df["duration_at_ss_2_hr"] = duration_2 * 5 / 60  # convert to hrs

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
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
