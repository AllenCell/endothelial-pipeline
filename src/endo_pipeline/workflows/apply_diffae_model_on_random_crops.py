from collections.abc import Sequence
from pathlib import Path

import fire
import pandas as pd
import torch
from cyto_dl.api import CytoDLModel

from cellsmap.util.manifest_preprocessing import save_file_to_fms
from src.endo_pipeline.configs import (
    DatasetConfig,
    ModelConfig,
    add_model_manifest,
    load_dataset_config,
    load_model_config,
    load_reference_dataset_configs,
    save_model_config,
)
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.model.apply_model import get_cytodl_commit_hash, load_overrides
from src.endo_pipeline.library.model.mlflow import download_model
from src.endo_pipeline.library.process.image_filepath_utils import extract_position_from_filepath

# the zarr creation workflow always has brightfield as channel index 1
ZARR_BF_CHANNEL = 1


def generate_overrides_for_random_crop_features(
    user_overrides: dict,
    save_path: str,
    data_path: str,
    ckpt_path: str,
    dataset_name: str,
    model_name: str,
) -> dict:
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
            "save_dir": save_path,
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


def generate_zarr_csv(
    dataset_config: DatasetConfig, save_path: Path, resolution_level: int = 0
) -> Path:
    """Generate a CSV file with path to Zarr files for the given dataset."""
    # generate csv with paths to zarr files
    df = pd.DataFrame({"path": dataset_config.zarr_path})
    df["channel"] = ZARR_BF_CHANNEL
    df["resolution"] = resolution_level
    data_path = save_path / "dataset.csv"
    df.to_csv(data_path, index=False)
    return data_path


def update_prediction_from_crops_with_metadata(
    dataset_name: str,
    model_name: str,
    crop_size: Sequence[int],
    mlflow_id: str,
    save_path: Path,
) -> Path:
    """Update the prediction file with metadata."""
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

    pred_df["position"] = pred_df["filename_or_obj"].apply(
        lambda s: extract_position_from_filepath(s, int_only=False)
    )
    pred_df.rename(columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True)
    pred_df.to_parquet(prediction_path)
    return prediction_path


def apply_model_single(
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
    elif isinstance(save_path, str):
        save_path = Path(save_path)

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    # create zarr dataset
    data_path = generate_zarr_csv(dataset_config, save_path, resolution_level)

    # apply overrides
    overrides = generate_overrides_for_random_crop_features(
        overrides,
        save_path=str(save_path),
        data_path=str(data_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
    )
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
        # note that this function will be deprecated in the future
        file_id = save_file_to_fms(
            str(prediction_path),
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


def main(
    model_name: str,
    dataset_names: str | Sequence[str] = "reference",
    resolution_level: int = 0,
    upload_to_fms: bool = True,
    save_path: str | Path | None = None,
    overrides: str | dict | None = None,
) -> None:
    """
    Apply a model to a multiple datasets.

    Example usage:
    ```
    uv run src/endo_pipeline/workflows/apply_model.py
    --model_name diffae_04_10 --dataset_names '["20241016_20X","20250224_20X"]'
    ```


    Parameters
    ----------
    model_name: str
        Name of the model from `model_config.yaml` to apply.
    dataset_names: str
        Names of the datasets from `data_config.yaml` to apply the model to.
        If "reference", all reference datasets will be used.
    resolution_level: int
        Resolution level to apply the model at. Default is 0 (highest resolution)
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str | Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    overrides: str or dict or None
        Overrides to apply to the model config. By default, no overrides are applied
    """
    # default is to apply to all reference datasets
    if dataset_names == "reference":
        dataset_config_list = load_reference_dataset_configs()
    else:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_names]

    # load model config
    model_config = load_model_config(model_name)

    # apply model to each dataset
    for dataset_config in dataset_config_list:
        model_config = apply_model_single(
            model_config=model_config,
            dataset_config=dataset_config,
            resolution_level=resolution_level,
            upload_to_fms=upload_to_fms,
            save_path=save_path,
            overrides=overrides,
        )

    # save out updated model config
    save_model_config(model_config)


if __name__ == "__main__":
    fire.Fire(main)
