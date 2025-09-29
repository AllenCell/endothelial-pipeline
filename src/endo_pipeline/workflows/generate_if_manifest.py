import argparse

from endo_pipeline.library.process.if_feature_extraction import run_nuclei_feature_extraction
from endo_pipeline.library.process.if_manifest import (
    save_manifest_to_csv,
    update_dataframe_manifest,
    upload_manifest_to_fms,
)

"""
Workflow to generate an immunofluorescence manifest for a given dataset.

To test this workflow, you can run the following command:
python src/endo_pipeline/workflows/generate_if_manifest.py --testing 20250509_20X_IF1

To run the workflow:
python src/endo_pipeline/workflows/generate_if_manifest.py 20250509_20X_IF2
"""


def main(datasets: list[str], testing: bool = False) -> None:
    """Run the main workflow function.

    Args:
        datasets (list[str]): List of dataset names to process.
        testing (bool): If True, only run feature extraction (Step 1) and skip steps 2-4.
    """
    for dataset in datasets:
        print(f"Starting workflow for dataset: {dataset}")

        # Step 1: Run feature extraction
        df = run_nuclei_feature_extraction(dataset)

        # Step 2: Save to CSV
        save_path = save_manifest_to_csv(dataset, df)

        if testing:
            print("Testing mode enabled. Skipping fms upload and dataset config update")
            continue

        # Step 3: Upload to FMS
        fms_id = upload_manifest_to_fms(save_path, dataset)

        # Step 4: Update dataset configuration
        update_dataframe_manifest(dataset, fms_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the immunofluorescence manifest workflow.")
    parser.add_argument(
        "datasets",
        nargs="+",  # Accept one or more datasets as arguments
        help="Dataset names to process.",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Enable testing mode to skip fms upload and dataset config update",
    )
    args = parser.parse_args()
    main(args.datasets, testing=args.testing)
