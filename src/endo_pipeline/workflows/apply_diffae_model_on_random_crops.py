from collections.abc import Sequence
from pathlib import Path

import fire
import torch
from cyto_dl.api import CytoDLModel

from src.endo_pipeline.configs import (
    DatasetConfig,
    ModelConfig,
    add_model_manifest,
    load_dataset_config,
    load_model_config,
    load_reference_dataset_configs,
    save_model_config,
)
from src.endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
from src.endo_pipeline.library.model.apply_model import (
    apply_model_on_one_dataset,
    get_cytodl_commit_hash,
    load_overrides,
)
from src.endo_pipeline.library.model.mlflow import download_model
from src.endo_pipeline.library.model.model_inputs import (
    generate_overrides_for_model_eval,
    generate_zarr_csv,
)
from src.endo_pipeline.library.model.model_outputs import update_prediction_from_crops_with_metadata


def _apply_model_single(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    resolution_level: int = 1,
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
        Resolution level to apply the model at. Default is 1 (zarr sample resolution)
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
    overrides = generate_overrides_for_model_eval(
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
    resolution_level: int = 1,
    upload_to_fms: bool = True,
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
        Resolution level to apply the model at. Default is 1 (zarr sample resolution).
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
        model_config = apply_model_on_one_dataset(
            model_config=model_config,
            dataset_config=dataset_config,
            resolution_level=resolution_level,
            upload_to_fms=upload_to_fms,
            overrides=overrides,
        )

    # save out updated model config
    save_model_config(model_config)


if __name__ == "__main__":
    fire.Fire(main)
