import argparse

import pandas as pd

from cellsmap.util.manifest_preprocessing.fms_upload import save_file_to_fms
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import load_single_dataset_config, save_dataset_config
from src.endo_pipeline.configs.dataset_io import get_git_versioning_info
from src.endo_pipeline.library.process.if_feature_extraction import run_nuclei_feature_extraction

"""
Workflow to generate an immunofluorescence manifest for a given dataset.

To test this workflow, you can run the following command:
python src/endo_pipeline/workflows/generate_if_manifest.py --testing 20250509_20X_IF1

To run the workflow:
python src/endo_pipeline/workflows/generate_if_manifest.py 20250509_20X_IF2

Current datasets to choose from:
20250509_20X_IF1 20250509_20X_IF2 20250509_20X_IF3 20250509_20X_IF4 20250509_20X_IF5
20250509_20X_IF6 20250509_20X_IF7 20250509_20X_IF8 20250509_20X_IF9 20250509_20X_IF10
20250509_20X_IF11 20250509_20X_IF12
"""


def save_manifest_to_csv(dataset: str, df: pd.DataFrame) -> str:
    """Save the extracted features to a CSV file.

    Args:
        dataset (str): The name of the dataset.
        df (pd.DataFrame): The DataFrame containing the extracted features.

    Returns:
        str: The path to the saved CSV file.
    """
    output_dir = get_output_path("immunoflouresence_manifest", verbose=True)
    save_path = output_dir + f"{dataset}_if_manifest.csv"
    df.to_csv(save_path, index=False)
    return save_path


def upload_manifest_to_fms(save_path: str, dataset: str) -> str:
    """Upload the manifest to FMS and return the FMS ID.

    Args:
        save_path (str): The path to the saved CSV file.
        dataset (str): The name of the dataset.

    Returns:
        str: The FMS ID of the uploaded file.
    """
    commit_info = get_git_versioning_info()
    fms_id = save_file_to_fms(
        file_path=save_path,
        dataset=dataset,
        commit_hash=commit_info["git_commit_hash"],
        misc_notes=f"This immunoflourescence manifest was produced by the cellsmap repository. \
                Made on branch {commit_info['git_branch_name']} at {commit_info['timestamp']}.",
        file_type="csv",
        model_version="",
        mlflow_run_id=None,
        effort="Core",
        env="prod",
    )
    return fms_id


def update_dataset_config(dataset: str, fms_id: str) -> None:
    """Update the dataset configuration with the FMS ID.

    Args:
        dataset (str): The name of the dataset.
        fms_id (str): The FMS ID of the uploaded file.

    Raises:
        ValueError: If the dataset configuration cannot be loaded.
    """
    dataset_config = load_single_dataset_config(dataset)
    dataset_config.immunofluorescence_manifest_fmsid = fms_id
    save_dataset_config(dataset_config)


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
        update_dataset_config(dataset, fms_id)


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
