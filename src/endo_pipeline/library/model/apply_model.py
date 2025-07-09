import json
from collections.abc import Callable
from pathlib import Path

import torch
from cyto_dl.api import CytoDLModel

from cellsmap.util.manifest_preprocessing import save_file_to_fms
from src.endo_pipeline.configs import DatasetConfig, ModelConfig, add_model_manifest
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.model.mlflow import download_mlflow_artifact, download_model


def get_cytodl_commit_hash(run_id: str, model_path: Path) -> str:
    """
    Extract commit hash from the requirements file uploaded to mlflow.

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

    with open(model_path / artifact_path) as f:
        lines = f.readlines()
    for line in lines:
        if "git+" in line and "cyto-dl" in line:
            commit_hash = line.split("git+")[1].split("#egg")[0].split("/")[-1]
            return commit_hash
    raise ValueError("No commit hash found in requirements.txt")


def load_overrides(overrides: str | dict | None) -> dict:
    """
    Load overrides from a string or dictionary.

    If None, return an empty dictionary.
    """
    if isinstance(overrides, str):
        overrides_dict = json.loads(overrides)
    elif overrides is None:
        overrides_dict = {}
    elif isinstance(overrides, dict):
        overrides_dict = overrides
    elif not isinstance(overrides, dict):
        raise ValueError("Overrides must be a dictionary or a string")
    return overrides_dict


def apply_model_single(
    generate_overrides: Callable,
    update_prediction_with_metadata: Callable,
    generate_dataframe_for_prediction: Callable,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    resolution_level: int = 0,
    upload_to_fms: bool = True,
    save_path: str | Path | None = None,
    overrides: str | dict | None = None,
) -> ModelConfig:
    """
    Apply a DiffAE model to a single dataset.

    Parameters
    ----------
    model_config: ModelConfig
        Configuration of the model to apply.
    dataset_config: DatasetConfig
        Configuration of the dataset to apply the model to.
    resolution_level: int
        Resolution level to apply the model at. Default is 0 (highest resolution)
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str or Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    overrides: str or dict or None
        Overrides to apply to the model config. By default, no overrides are applied
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    overrides = load_overrides(overrides)
    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, include_timestamp=False)
    path_dict = download_model(mlflow_id, model_path)

    if save_path is None:
        # if no save path is provided, use the default path
        save_path = get_output_path(
            "models", model_config.name, dataset_config.name, include_timestamp=False
        )

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    # create dataframe (saved as csv)
    # to pass to the model for prediction
    data_path = generate_dataframe_for_prediction(dataset_config, save_path, resolution_level)

    # apply overrides
    overrides = generate_overrides(
        overrides,
        save_path=save_path,
        data_path=data_path,
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
    )
    model.override_config(overrides)
    model.predict()
    crop_size = model.cfg.model.spatial_inferer.splitter.patch_size

    prediction_path = update_prediction_with_metadata(
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        crop_size=crop_size,
        mlflow_id=mlflow_id,
        save_path=save_path,
    )

    if upload_to_fms:
        # note that this function will be deprecated in the future
        file_id = save_file_to_fms(
            prediction_path,
            dataset_config.name,
            get_cytodl_commit_hash(mlflow_id, model_path),
            misc_notes="",
            mlflow_run_id=mlflow_id,
        )

        # add new manifest to model config
        model_config = add_model_manifest(
            model_config,
            dataset_config.name,
            file_id,
        )

    return model_config
