from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fire
import pandas as pd
import torch
from cyto_dl.api import CytoDLModel

from cellsmap.util.manifest_io import get_dataframe_by_fmsid
from cellsmap.util.manifest_preprocessing import save_file_to_fms
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import load_single_dataset_config, save_dataset_config
from src.endo_pipeline.configs.dataset_io import extract_P, get_model_info
from src.endo_pipeline.library.model.diffae.apply_diffae_model import (
    get_cytodl_commit_hash,
    load_overrides,
)
from src.endo_pipeline.library.model.mlflow import download_model

ZARR_BF_CHANNEL = 1


def generate_overrides(
    user_overrides: dict[str, Any],
    save_path: str,
    data_path: str,
    ckpt_path: str,
    dataset_name: str,
    model_name: str,
) -> dict[str, Any]:
    """Generate overrides for the CytoDLModel configuration."""
    overrides = {
        # train and val dataloaders are unnecessary for prediction
        # and might be slow to instantiate (e.g. if they cache data)
        "data.train_dataloaders": None,
        "data.val_dataloaders": None,
        "data.predict_dataloaders.num_workers": 128,
        "data.predict_dataloaders.dataset.csv_path": data_path,
        "paths.output_dir": save_path,
        # change checkpoint path to the one downloaded from mlflow
        "checkpoint.ckpt_path": ckpt_path,
        "checkpoint.strict": True,
        "callbacks": None,
        "callbacks.prediction_saver": {
            "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
            "save_dir": str(save_path),
            "meta_keys": [
                "T",
                "start_y",
                "start_x",
                "end_y",
                "end_x",
                "filename_or_obj",
                "track_id",
            ],
            "save_suffix": f"{dataset_name}_{model_name}_crop_features",
        },
        # add cropping transform
        "data.predict_dataloaders.dataset.transform.transforms[6]": {
            "_target_": "cyto_dl.image.transforms.coordinate_crop.CropToCoordsd",
            "keys": ["raw_bf"],
            "start_keys": ["start_y", "start_x"],
            "end_keys": ["end_y", "end_x"],
            "meta_keys": ["track_id"],
        },
        # persist coordinate data through MultiDimImageDataset
        "data.predict_dataloaders.dataset.extra_columns": [
            "start_y",
            "start_x",
            "end_y",
            "end_x",
            "track_id",
        ],
        # no spatial inferer needed
        "model.spatial_inferer": None,
    }
    overrides.update(user_overrides)
    return overrides


def centroid_to_bbox(df: pd.DataFrame):
    """
    Convert centroids to bounding boxes.

    Note: coordinates are downsampled by half to match current model resolution.
    """
    df["start_x"] = ((df["centroid_x"] - df["crop_size"] / 2) / 2).astype(int)
    df["start_y"] = ((df["centroid_y"] - df["crop_size"] / 2) / 2).astype(int)
    df["end_x"] = ((df["centroid_x"] + df["crop_size"] / 2) / 2).astype(int)
    df["end_y"] = ((df["centroid_y"] + df["crop_size"] / 2) / 2).astype(int)
    return df


def preprocess_manifest(dataset_name: str, save_dir: str) -> str:
    """Preprocess the manifest for a dataset to prepare it for model prediction."""
    fms_id = load_single_dataset_config(dataset_name).tracking_integration_fmsid
    df = get_dataframe_by_fmsid(fms_id)
    # convert centroids to bounding boxes
    df = centroid_to_bbox(df)

    # group df by zarr_path and convert start and end coordinates to list
    grouped_df = (
        df.groupby(["zarr_path", "image_index"])
        .agg(
            {
                "start_y": lambda x: list(x),
                "start_x": lambda x: list(x),
                "end_y": lambda x: list(x),
                "end_x": lambda x: list(x),
                "track_id": lambda x: list(x),
            }
        )
        .reset_index()
    )
    grouped_df["channel"] = ZARR_BF_CHANNEL
    grouped_df["resolution"] = 0
    # only run a single timepoint from zarr
    grouped_df["start"] = grouped_df["image_index"]
    grouped_df["stop"] = grouped_df["image_index"]
    grouped_df.rename({"zarr_path": "path", "image_index": "T"}, axis=1, inplace=True)

    save_path = save_dir / "aggregated_crop_manifest.csv"
    grouped_df.to_csv(save_path, index=False)
    return save_path


def update_prediction_with_meta(
    dataset_name: str, model_name: str, mlflow_id: str, save_path: Path
):
    """Update the prediction file with metadata."""
    # add model and dataset information to prediction file
    prediction_path = save_path / f"predict_{dataset_name}_{model_name}_crop_features.parquet"
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


def apply_model_single(
    model_name: str,
    dataset_name: str,
    save_path: str | Path | None = None,
    upload_to_fms: bool = True,
    overrides: str | dict | None = None,
):
    """Apply a model to a single dataset."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    overrides = load_overrides(overrides)
    # download model from mlflow
    mlflow_id = get_model_info(model_name)["mlflow_run_id"]
    model_path = Path(get_output_path(f"models/{model_name}"))
    path_dict = download_model(mlflow_id, model_path)

    save_path = save_path or model_path / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    data_path = preprocess_manifest(dataset_name, save_path)

    # apply overrides
    overrides = generate_overrides(
        overrides,
        save_path=save_path,
        data_path=data_path,
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_name,
        model_name=model_name,
    )
    model.override_config(overrides)
    model.predict()

    prediction_path = update_prediction_with_meta(
        dataset_name=dataset_name,
        model_name=model_name,
        mlflow_id=mlflow_id,
        save_path=save_path,
    )
    commit_hash = get_cytodl_commit_hash(mlflow_id, model_path)

    if upload_to_fms:
        file_id = save_file_to_fms(
            prediction_path,
            dataset_name,
            commit_hash,
            misc_notes="",
            mlflow_run_id=mlflow_id,
        )

        # update dataset config with the FMS ID
        # of the prediction file
        dataset_config = load_single_dataset_config(dataset_name)
        dataset_config.diffae_tracking_integration_fmsid = file_id
        save_dataset_config(dataset_config)

    return prediction_path


def apply_model(
    model_name: str,
    dataset_names: Sequence[str],
    upload_to_fms: bool = True,
    save_path: str | Path | None = None,
    overrides: str | dict | None = None,
):
    """
    Apply a model to a multiple datasets.

    Example usage:
    ```
    python src/endo_pipeline/workflows/apply_on_crop.py
    --model_name diffae_04_10 --dataset_names '["20241016_20X","20250224_20X"]'
    ```


    Parameters
    ----------
    model_name: str
        Name of the model from `model_config.yaml` to apply.
    dataset_names: str
        Name of the dataset from `data_config.yaml` to apply the model to.
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str | Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    overrides: str or dict or None
        Overrides to apply to the model config. By default, no overrides are applied
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    for name in dataset_names:
        apply_model_single(
            model_name=model_name,
            dataset_name=name,
            upload_to_fms=upload_to_fms,
            save_path=save_path,
            overrides=overrides,
        )


if __name__ == "__main__":
    fire.Fire(apply_model)
