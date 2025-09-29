from pathlib import Path

import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
from endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    load_dataframe_manifest,
    save_dataframe_manifest,
)


def save_manifest_to_csv(dataset: str, df: pd.DataFrame) -> Path:
    """Save the extracted features to a CSV file.

    Args:
        dataset (str): The name of the dataset.
        df (pd.DataFrame): The DataFrame containing the extracted features.

    Returns:
        str: The path to the saved CSV file.
    """
    output_dir = get_output_path("immunofluorescence_manifest")
    save_path = output_dir / f"{dataset}_if_manifest.csv"
    df.to_csv(save_path, index=False)
    return save_path


def upload_manifest_to_fms(save_path: Path, dataset: str) -> str:
    """Upload the manifest to FMS and return the FMS ID.

    Args:
        save_path (str): The path to the saved CSV file.
        dataset (str): The name of the dataset.

    Returns:
        str: The FMS ID of the uploaded file.
    """

    dataset_config = load_dataset_config(dataset)
    annotations = build_fms_annotations(dataset_config)
    fms_id = upload_file_to_fms(save_path, annotations, "csv")

    return fms_id


def update_dataframe_manifest(dataset: str, fms_id: str) -> None:
    """Update the dataframe manifest with the FMS ID.

    Args:
        dataset (str): The name of the dataset.
        fms_id (str): The FMS ID of the uploaded file.

    Raises:
        ValueError: If the dataset configuration cannot be loaded.
    """

    try:
        manifest = load_dataframe_manifest("immunofluorescence")
    except FileNotFoundError:
        manifest = DataframeManifest(name="immunofluorescence", workflow=Path(__file__).stem)

    manifest.locations[dataset] = DataframeLocation(fmsid=fms_id)
    save_dataframe_manifest(manifest)
