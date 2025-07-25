import json
from pathlib import Path

import torch
from cyto_dl.api import CytoDLModel

from src.endo_pipeline.configs import (
    CytoDLModelConfig,
    DatasetConfig,
    add_model_manifest,
    save_dataset_config,
)
from src.endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms

from .mlflow_utils import download_mlflow_artifact, download_model
from .model_inputs import (
    generate_overrides_for_model_eval,
    generate_overrides_for_track_based_crops,
    generate_zarr_csv_for_model_eval,
    preprocess_tracking_manifest_for_model_eval,
)
from .model_outputs import (
    update_prediction_from_crops_with_metadata,
    update_prediction_from_tracks_with_metadata,
)


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
        artifact_path = Path("requirements/train-requirements.txt")
        download_mlflow_artifact(run_id, artifact_path, model_path)
    except ValueError:
        artifact_path = Path("requirements/eval-requirements.txt")
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


def apply_model_on_grid_of_crops_from_one_dataset(
    model_config: CytoDLModelConfig,
    dataset_config: DatasetConfig,
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
    z_stack_offsets: tuple[int, int] | None = None,
) -> CytoDLModelConfig:
    """
    Apply a DiffAE model to a single dataset.

    Parameters
    ----------
    model_config: CytoDLModelConfig
        Configuration of the model to apply.
    dataset_config: DatasetConfig
        Configuration of the dataset to apply the model to.
    resolution_level: int
        Resolution level to apply the model at. Default is 1 (zarr sample resolution)
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str or Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    user_overrides: str or dict or None
        Additional overrides to apply to the model config. By default, no overrides are applied.
    z_stack_offsets: tuple[int, int] | None
        If None, all z-slices are loaded. Default is None.
        If provided, limits the number of z-slices to load from the raw brightfield images.
        First element is the lower offset, how many slices below the center plane to include, and
        the second element is the upper offset, how many slices above the center plane to include.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    user_overrides_ = load_overrides(user_overrides)
    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, "train", include_timestamp=False)
    path_dict = download_model(mlflow_id, model_path)

    # set default output path
    save_path = get_output_path(
        "models", model_config.name, dataset_config.name, include_timestamp=False
    )

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    # create csv with zarr paths and args for loading and processing images
    data_path = generate_zarr_csv_for_model_eval(
        dataset_config, save_path, resolution_level, z_stack_offsets
    )

    # apply overrides for model evaluation
    overrides = generate_overrides_for_model_eval(
        user_overrides=user_overrides_,
        save_path=str(save_path),
        data_path=str(data_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        z_stack_offsets=z_stack_offsets,
    )

    # override model config with the overrides
    model.override_config(overrides)
    model.predict()
    crop_size = model.cfg.model.spatial_inferer.splitter.patch_size

    prediction_path = update_prediction_from_crops_with_metadata(
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        crop_size=crop_size,
        mlflow_id=mlflow_id,
        save_path=save_path,
    )

    if upload_to_fms:
        # build FMS annotations
        dataset_annotations = build_fms_annotations(
            dataset_config,
            include_timestamp=False,
            include_git_info=False,
            model=model_config,
        )

        # upload prediction file to FMS and get file ID
        file_id = upload_file_to_fms(
            prediction_path,
            annotations=dataset_annotations,
            file_type="parquet",
        )

        # add new manifest to model config
        model_config = add_model_manifest(
            model_config, dataset_config.name, file_id, z_stack_offsets=z_stack_offsets
        )

    return model_config


def apply_model_on_tracked_crops_from_one_dataset(
    model_config: CytoDLModelConfig,
    dataset_config: DatasetConfig,
    save_path: str | Path | None = None,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
) -> None:
    """
    Apply a DiffAE model to a single dataset with
    cell segmentation and tracking.

    Parameters
    ----------
    model_config: CytoDLModelConfig
        Configuration of the model to apply.
    dataset_config: DatasetConfig
        Configuration of the dataset to apply the model to.
    resolution_level: int
        Resolution level to apply the model at. Default is 0 (highest resolution)
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str or Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    user_overrides: str or dict or None
        Additional overrides to apply to the model config. By default, no overrides are applied
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    user_overrides_ = load_overrides(user_overrides)
    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, include_timestamp=False)
    path_dict = download_model(mlflow_id, model_path)

    if save_path is None:
        # if no save path is provided, use the default path
        save_path = get_output_path(
            "models", model_config.name, dataset_config.name, include_timestamp=False
        )
    elif isinstance(save_path, str):
        save_path = Path(save_path)

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    # process tracking manifest for model evaluation
    # this is used for loading and processing images
    data_path = preprocess_tracking_manifest_for_model_eval(dataset_config, save_path)

    # apply overrides for model evaluation on tracked crops
    overrides = generate_overrides_for_track_based_crops(
        user_overrides=user_overrides_,
        save_path=str(save_path),
        data_path=str(data_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
    )
    model.override_config(overrides)
    model.predict()

    prediction_path = update_prediction_from_tracks_with_metadata(
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        mlflow_id=mlflow_id,
        save_path=save_path,
    )

    if upload_to_fms:
        # build FMS annotations
        dataset_annotations = build_fms_annotations(
            dataset_config,
            include_timestamp=False,
            include_git_info=False,
            model=model_config,
        )

        # upload prediction file to FMS and get file ID
        file_id = upload_file_to_fms(
            prediction_path,
            annotations=dataset_annotations,
            file_type="parquet",
        )

        # tracking integration FMS ID
        # is stored in the dataset config
        dataset_config.diffae_tracking_integration_fmsid = file_id
        save_dataset_config(dataset_config)
