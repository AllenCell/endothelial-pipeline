import datetime
from pathlib import Path
from typing import Literal

import fire
from cyto_dl.api import CytoDLModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.endo_pipeline.configs import ModelConfig, get_config_dir, save_model_config
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.model import (
    generate_overrides_for_model_training,
    get_dataset_names_used_for_training,
)


def _initialize_diffae_model(
    training_config: DictConfig | ListConfig,
    crop_size: int,
    model_name: str,
    train_csv_path: Path,
    val_csv_path: Path,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    Parameters
    ----------
    training_config: DictConfig | ListConfig
        The training configuration to use.
    crop_size: int
        The pixel size of the image crop to use for training.
        This is the crop size along one dimension, the image will be square
        with size (crop_size, crop_size).
    model_name: str
        The name of the model to train.
    train_csv_path: Path | None
        The path to the training dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.
    val_csv_path: Path | None
        The path to the validation dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.
    """
    # user overrides for training
    overrides = generate_overrides_for_model_training(
        model_name, crop_size, train_csv_path, val_csv_path
    )

    # init model
    model = CytoDLModel()
    # override config with workflow inputs
    model.load_config_from_dict(training_config)
    model.override_config(overrides)
    return model


def _get_valid_csv_path(csv_path: Path | str | None, csv_name: Literal["train", "val"]) -> Path:
    """
    Get a valid CSV path for training or validation datasets.

    Parameters
    ----------
    csv_path: Path | str | None
        The path to the CSV file. If None, the default path for the output of
        generate_csv_for_training_diffae will be used.
    csv_name: Literal["train", "val"]
        The name of the CSV file to validate. If csv_path is not None,
        csv_name will not be used in the path generation.
        This input is mainly for the default case where csv_path is None,
        and the path will be generated based on the csv_name (train or val).


    Returns
    -------
    Path
        A valid Path object pointing to the CSV file.
    """
    if csv_path is None:
        csv_path = get_output_path("manifests", include_timestamp=False) / f"{csv_name}.csv"

    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please provide a valid path.")

    return csv_path


def main(
    crop_size: int = 128,
    train_csv_path: Path | str | None = None,
    val_csv_path: Path | str | None = None,
) -> None:
    """
    Train a DiffAE model using the provided configuration.

    Parameters
    ----------
    crop_size: int
        The pixel size of the image crop to use for training. Default is 128.
        This is the crop size along one dimension, the image will be square
        with size (crop_size, crop_size).

    train_csv_path: Path | None
        The path to the training dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.

    val_csv_path: Path | None
        The path to the validation dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.
    """
    # get valid CSV paths for training and validation datasets
    train_csv_path = _get_valid_csv_path(train_csv_path, "train")
    val_csv_path = _get_valid_csv_path(val_csv_path, "val")

    # load training config
    training_config = OmegaConf.load(get_config_dir() / "train_diffae.yaml")

    # set model name via timestamp and crop size
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"diffae_patch_{crop_size}x{crop_size}_{timestamp}"

    # initialize DiffAE model: generates config
    # overrides and sets up output directories
    model = _initialize_diffae_model(
        training_config,
        crop_size,
        model_name,
        train_csv_path,
        val_csv_path,
    )
    _, object_dict = model.train()

    # retrive MLflow run ID
    mlflow_logger = object_dict["logger"][0]
    run_id = mlflow_logger.run_id
    # get list of datasets used for training
    list_of_training_datasets = get_dataset_names_used_for_training(
        train_csv_path,
        val_csv_path,
        "live_20X_objective_3i_microscope",
    )
    # add run ID to model config
    model_config = ModelConfig(
        name=model_name,
        mlflow_run_id=run_id,
        training_datasets=list_of_training_datasets,
    )
    # save the model config
    save_model_config(model_config)


if __name__ == "__main__":
    fire.Fire(main)
