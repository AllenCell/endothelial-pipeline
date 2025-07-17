import datetime
import os
from pathlib import Path

import fire
from cyto_dl.api import CytoDLModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.endo_pipeline.configs import ModelConfig, get_config_dir, save_model_config
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.model import get_dataset_names_used_for_training


def _generate_training_overrides(
    model_name: str,
    crop_size: int,
    train_csv_path: Path | None = None,
    val_csv_path: Path | None = None,
) -> dict:
    """
    Generate overrides for the DiffAE model training configuration.

    Parameters
    ----------
    model_name: str
        The name of the model to train.

    crop_size: int
        The number of pixels in each dimension of the
        image crop to use for training.

        That is, the cropped image will be square
        with size (crop_size px, crop_size px).

    train_csv_path: Path | None
        The path to the training dataset CSV file.
        If None, the default path for the output of
        generate_csv_for_training_diffae will be used.

    val_csv_path: Path | None
        The path to the validation dataset CSV file.
        If None, the default path for the output of
        generate_csv_for_training_diffae will be used.
    """
    # create output directories if they do not exist
    train_output_path = get_output_path("models", model_name, "train", include_timestamp=False)
    _ = get_output_path("models", model_name, "train", "logs", include_timestamp=False)
    _ = get_output_path("models", model_name, "train", "checkpoints", include_timestamp=False)

    if train_csv_path is None:
        # use default path for training CSV if not provided
        train_csv_path = get_output_path("manifests", include_timestamp=False) / "train.csv"
        if not train_csv_path.exists():
            raise FileNotFoundError(
                f"Training CSV file not found at {train_csv_path}. "
                "Please provide a valid path or generate the CSV file."
            )
    if val_csv_path is None:
        # use default path for validation CSV if not provided
        val_csv_path = get_output_path("manifests", include_timestamp=False) / "val.csv"
        if not val_csv_path.exists():
            raise FileNotFoundError(
                f"Validation CSV file not found at {val_csv_path}. "
                "Please provide a valid path or generate the CSV file."
            )

    overrides = {
        # set path to train and val datasets
        "data.train_dataloaders.dataset.csv_path": train_csv_path.as_posix(),
        "data.predict_dataloaders.dataset.csv_path": val_csv_path.as_posix(),
        "data.val_dataloaders.dataset.csv_path": val_csv_path.as_posix(),
        # get repo root directory and current working directory
        "paths.root_dir": Path(__file__).resolve().parents[3],
        "paths.work_dir": os.getcwd(),
        # save outputs to user-specified directory
        "paths.output_dir": (train_output_path / "logs").as_posix(),
        "paths.log_dir": "${paths.output_dir}",
        "callbacks.model_checkpoint.dirpath": (train_output_path / "checkpoints").as_posix(),
        # update run name
        "run_name": model_name,
        # set crop size from input via model.image_shape,
        # the rest are populated by interpolation
        "model.image_shape": [1, crop_size, crop_size],
    }
    return overrides


def _initialize_diffae_model(
    training_config: DictConfig | ListConfig,
    crop_size: int,
    model_name: str,
    train_csv_path: Path | None = None,
    val_csv_path: Path | None = None,
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
    overrides = _generate_training_overrides(model_name, crop_size, train_csv_path, val_csv_path)

    # init model
    model = CytoDLModel()
    # override config with workflow inputs
    model.load_config_from_dict(training_config)
    model.override_config(overrides)
    return model


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
    # convert paths to Path objects if they are strings
    if isinstance(train_csv_path, str):
        train_csv_path = Path(train_csv_path)
    if isinstance(val_csv_path, str):
        val_csv_path = Path(val_csv_path)

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
    list_of_training_datasets = get_dataset_names_used_for_training(train_csv_path, val_csv_path)
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
