from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fire
import pandas as pd
import torch
from cyto_dl.api import CytoDLModel

from src.endo_pipeline.configs import (
    DatasetConfig,
    ModelConfig,
    load_dataset_config,
    load_model_config,
    save_dataset_config,
)
from src.endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe_from_fms,
    upload_file_to_fms,
)
from src.endo_pipeline.library.model.apply_model import get_cytodl_commit_hash, load_overrides
from src.endo_pipeline.library.model.mlflow import download_model
from src.endo_pipeline.library.model.model_inputs import (
    generate_overrides_for_track_based_crops,
    preprocess_tracking_manifest_for_model_eval,
)
from src.endo_pipeline.library.process.image_filepath_utils import extract_position_from_filepath


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
    pred_df["position"] = pred_df["filename_or_obj"].apply(
        lambda s: extract_position_from_filepath(s, int_only=False)
    )
    pred_df.rename(columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True)
    pred_df.to_parquet(prediction_path)
    return prediction_path


def apply_model_single(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    save_path: str | Path | None = None,
    upload_to_fms: bool = True,
    overrides: str | dict | None = None,
) -> None:
    """
    Apply a DiffAE model to a single dataset with
    cell segmentation and tracking.

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

    data_path = preprocess_tracking_manifest_for_model_eval(dataset_config, save_path)

    # apply overrides
    overrides = generate_overrides_for_track_based_crops(
        overrides,
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
            additional_notes=get_cytodl_commit_hash(mlflow_id, model_path),
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


def main(
    model_name: str,
    dataset_names: Sequence[str],
    upload_to_fms: bool = True,
    save_path: str | Path | None = None,
    overrides: str | dict | None = None,
) -> None:
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
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_names]

    # load model config
    model_config = load_model_config(model_name)

    # apply model to each dataset
    for dataset_config in dataset_config_list:
        apply_model_single(
            model_config=model_config,
            dataset_config=dataset_config,
            upload_to_fms=upload_to_fms,
            save_path=save_path,
            overrides=overrides,
        )


if __name__ == "__main__":
    fire.Fire(main)
