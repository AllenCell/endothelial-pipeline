from pathlib import Path
from typing import Sequence

import pandas as pd

from src.endo_pipeline.configs.dataset_io import extract_P

ZARR_BF_CHANNEL = 1  # Brightfield channel index for Zarr files


def update_prediction_from_crops_with_metadata(
    dataset_name: str,
    model_name: str,
    crop_size: Sequence[int],
    mlflow_id: str,
    save_path: Path,
) -> Path:
    """
    Update the prediction file with metadata,
    return the path to the updated prediction file.
    """
    # add model and dataset information to prediction file
    prediction_path = save_path / f"predict_{dataset_name}_{model_name}_features.parquet"
    pred_df = pd.read_parquet(prediction_path)
    pred_df["dataset"] = dataset_name
    pred_df["model_name"] = model_name
    pred_df["mlflow_id"] = mlflow_id

    # note: the current model loads images at resolution
    # level 0 and downsamples in the transforms.
    pred_df["resolution_level"] = 1

    pred_df["end_y"] = pred_df["start_y"] + crop_size[0]
    pred_df["end_x"] = pred_df["start_x"] + crop_size[1]
    pred_df["crop_size_y"] = crop_size[0]
    pred_df["crop_size_x"] = crop_size[1]

    pred_df["position"] = pred_df["filename_or_obj"].apply(lambda s: extract_P(s, int_only=False))
    pred_df.rename(columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True)
    pred_df.to_parquet(prediction_path)
    return prediction_path


def update_prediction_from_tracks_with_metadata(
    dataset_name: str, model_name: str, mlflow_id: str, save_path: Path
) -> Path:
    """Update the prediction file with metadata."""
    # add model and dataset information to prediction file
    prediction_path = (
        save_path / f"predict_{dataset_name}_{model_name}_tracked_crop_features.parquet"
    )
    pred_df = pd.read_parquet(prediction_path)
    pred_df["dataset"] = dataset_name
    pred_df["model_name"] = model_name
    pred_df["mlflow_id"] = mlflow_id

    # NOTE: the current model loads images at resolution level 0 and downsamples in the transforms.
    pred_df["resolution_level"] = 1

    crop_size = (
        pred_df["end_y"].iloc[0] - pred_df["start_y"].iloc[0],
        pred_df["end_x"].iloc[0] - pred_df["start_x"].iloc[0],
    )
    pred_df["crop_size_y"] = crop_size[0]
    pred_df["crop_size_x"] = crop_size[1]
    pred_df["position"] = pred_df["filename_or_obj"].apply(lambda s: extract_P(s, int_only=False))
    pred_df.rename(columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True)
    pred_df.to_parquet(prediction_path)
    return prediction_path
