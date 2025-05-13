import json
from pathlib import Path
from typing import Dict, Sequence, Union

import fire
import pandas as pd
import torch
from cyto_dl.api import CytoDLModel

from cellsmap.model_features.utils.mlflow_utils import (
    download_mlflow_artifact,
    download_model,
)
from cellsmap.util.dataset_io import (
    extract_P,
    get_model_info,
    get_zarr_path,
    update_dataset_config,
)
from cellsmap.util.manifest_preprocessing import save_file_to_fms
from cellsmap.util.set_output import get_output_path

# the zarr creation workflow always has brightfield as channel index 1
ZARR_BF_CHANNEL = 1


def get_cytodl_commit_hash(run_id: str, model_path: Path) -> str:
    """
    Extract commit hash from the requirements file uploaded to mlflow

    Parameters
    ----------
    run_id: str
        The run ID of the MLflow run.
    model_path: Path
        The path where the downloaded model artifacts are saved.
    """
    try:
        artifact_path = "requirements/train-requirements.txt"
        download_mlflow_artifact(run_id, artifact_path, model_path)
    except ValueError:
        artifact_path = "requirements/eval-requirements.txt"
        download_mlflow_artifact(run_id, artifact_path, model_path)

    with open(model_path / artifact_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "git+" in line and "cyto-dl" in line:
            commit_hash = line.split("git+")[1].split("#egg")[0].split("/")[-1]
            return commit_hash
    raise ValueError("No commit hash found in requirements.txt")


def generate_overrides(
    user_overrides,
    save_path: str,
    data_path: str,
    ckpt_path: str,
    dataset_name: str,
    model_name: str,
) -> Dict:
    overrides = {
        # train and val dataloaders are unnecessary for prediction and might be slow to instantiate (e.g. if they cache data)
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
                "filename_or_obj",
            ],
            "save_suffix": f"{dataset_name}_{model_name}_features",
        },
    }
    overrides.update(user_overrides)
    return overrides


def generate_zarr_csv(dataset_name: str, save_path: str, resolution_level: int = 0):
    # generate csv with paths to zarr files
    df = pd.DataFrame({"path": sorted(get_zarr_path(dataset_name).values())})
    df["channel"] = ZARR_BF_CHANNEL
    df["resolution"] = resolution_level
    data_path = str(save_path / "dataset.csv")
    df.to_csv(data_path, index=False)
    return data_path


def update_prediction_with_meta(
    dataset_name: str,
    model_name: str,
    crop_size: Sequence[int],
    mlflow_id: str,
    save_path: Path,
):
    # add model and dataset information to prediction file
    prediction_path = (
        save_path / f"predict_{dataset_name}_{model_name}_features.parquet"
    )
    pred_df = pd.read_parquet(prediction_path)
    pred_df["dataset"] = dataset_name
    pred_df["model_name"] = model_name
    pred_df["mlflow_id"] = mlflow_id

    # NOTE: the current model loads images at resolution level 0 and downsamples in the transforms.
    pred_df["resolution_level"] = 1

    pred_df["end_y"] = pred_df["start_y"] + crop_size[0]
    pred_df["end_x"] = pred_df["start_x"] + crop_size[1]
    pred_df["crop_size_y"] = crop_size[0]
    pred_df["crop_size_x"] = crop_size[1]
    pred_df["position"] = pred_df["filename_or_obj"].apply(
        lambda s: extract_P(s, int_only=False)
    )
    pred_df.rename(
        columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True
    )
    pred_df.to_parquet(prediction_path)
    return prediction_path


def load_overrides(overrides: Union[str, Dict]) -> Dict:
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    elif not isinstance(overrides, dict):
        raise ValueError("Overrides must be a dictionary or a string")
    return overrides


def apply_model_single(
    model_name: str,
    dataset_name: str,
    resolution_level: int = 0,
    upload_to_fms: bool = True,
    save_path: Union[str, Path] = None,
    overrides: Union[str, Dict] = {},
):
    """
    Apply a model to a single dataset.

    Parameters
    ----------
    model_name: str
        Name of the model from `model_config.yaml` to apply.
    dataset_name: str
        Name of the dataset from `data_config.yaml` to apply the model to.
    resolution_level: int
        Resolution level to apply the model at. Default is 0 (highest resolution)
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    overrides: str or dict
        Overrides to apply to the model config. By default, no overrides are applied
    """
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

    # create zarr dataset
    data_path = generate_zarr_csv(dataset_name, save_path, resolution_level)
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
    crop_size = model.cfg.model.spatial_inferer.splitter.patch_size

    prediction_path = update_prediction_with_meta(
        dataset_name=dataset_name,
        model_name=model_name,
        crop_size=crop_size,
        mlflow_id=mlflow_id,
        save_path=save_path,
    )

    if upload_to_fms:
        file_id = save_file_to_fms(
            prediction_path,
            dataset_name,
            get_cytodl_commit_hash(mlflow_id, model_path),
            misc_notes="",
            mlflow_run_id=mlflow_id,
        )

        update_dataset_config(
            dataset_name,
            {"diffae_manifest_fmsid": file_id},
        )

    return prediction_path


def apply_model(
    model_name: str,
    dataset_names: Sequence[str],
    resolution_level: int = 0,
    upload_to_fms: bool = True,
    save_path: Union[str, Path] = None,
    overrides: Union[str, Dict] = {},
):
    """
    Apply a model to a multiple datasets.
    Example usage: python apply_model.py --model_name diffae_04_10 --dataset_names '["20241016_20X","20250224_20X"]'


    Parameters
    ----------
    model_name: str
        Name of the model from `model_config.yaml` to apply.
    dataset_name: str
        Name of the dataset from `data_config.yaml` to apply the model to.
    resolution_level: int
        Resolution level to apply the model at. Default is 0 (highest resolution)
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    overrides: str or dict
        Overrides to apply to the model config. By default, no overrides are applied
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    for name in dataset_names:
        apply_model_single(
            model_name=model_name,
            dataset_name=name,
            resolution_level=resolution_level,
            upload_to_fms=upload_to_fms,
            save_path=save_path,
            overrides=overrides,
        )


if __name__ == "__main__":
    fire.Fire(apply_model)
