import datetime
from pathlib import Path

import fire
from cyto_dl.api import CytoDLModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.endo_pipeline.configs import get_config_dir, update_model_config_with_run_id
from src.endo_pipeline.io import get_output_path


def _generate_training_overrides(model_name: str, crop_size: int, save_path: Path) -> dict:
    """
    Generate overrides for the DiffAE model training configuration.

    Parameters
    ----------
    model_name: str
        The name of the model to train.
    crop_size: int
        The pixel size of the image crop to use for training.
        This is the crop size along one dimension, the image will be square
        with size (crop_size, crop_size).
    save_path: Path
        The path to the directory where the checkpoints and logs will be saved.
    """
    # create output directories if they do not exist
    for subdir in ["logs", "checkpoints"]:
        get_output_path("models", model_name, subdir, include_timestamp=False)

    manifest_path = get_output_path("manifests", include_timestamp=False)

    overrides = {
        # set path to train and val datasets
        "data.train_dataloaders.dataset.csv_path": (manifest_path / "train.csv").as_posix(),
        "data.predict_dataloaders.dataset.csv_path": (manifest_path / "val.csv").as_posix(),
        "data.val_dataloaders.dataset.csv_path": (manifest_path / "val.csv").as_posix(),
        # save outputs to user-specified directory
        "model.save_dir": (save_path / "logs").as_posix(),
        "trainer.default_root_dir": save_path,
        "callbacks.model_checkpoint.dirpath": (save_path / "checkpoints").as_posix(),
        "paths.root_dir": Path(__file__).resolve().parents[3],
        "paths.log_dir": (save_path / "logs").as_posix(),
        # make sure that last ckpt is saved
        "callbacks.model_checkpoint.monitor": None,
        # update run name
        "run_name": model_name,
        # set crop size from input
        "model.image_shape": [1, crop_size, crop_size],
        "model.spatial_inferer.splitter.patch_size": [crop_size, crop_size],
    }
    return overrides


def _initialize_diffae_model(
    training_config: DictConfig | ListConfig,
    crop_size: int,
    model_name: str,
    save_path: Path,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    Parameters
    ----------
    training_config: OmegaConf
        The training configuration to use.
    crop_size: int
        The pixel size of the image crop to use for training.
        This is the crop size along one dimension, the image will be square
        with size (crop_size, crop_size).
    model_name: str
        The name of the model to train.
    save_path: Path
        The path to the directory where the checkpoints and logs will be saved.
    """
    # user overrides for training
    overrides = _generate_training_overrides(model_name, crop_size, save_path)

    # init model
    model = CytoDLModel()
    # override config with workflow inputs
    model.load_config_from_dict(training_config)
    model.override_config(overrides)
    return model


def main(crop_size: int = 128) -> None:
    """
    Train a DiffAE model using the provided configuration.

    Parameters
    ----------
    crop_size: int
        The pixel size of the image crop to use for training. Default is 128.
        This is the crop size along one dimension, the image will be square
        with size (crop_size, crop_size).
    """
    # load training config
    training_config = OmegaConf.load(get_config_dir() / "train_diffae.yaml")

    # set model name via timestamp and crop size
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    model_name = f"diffae_patch_{crop_size}x{crop_size}_{timestamp}"
    # set save directory
    save_path = get_output_path("models", model_name, include_timestamp=False)

    model = _initialize_diffae_model(
        training_config,
        crop_size,
        model_name,
        save_path,
    )
    _, object_dict = model.train()

    # retrive MLflow run ID
    mlflow_logger = object_dict["logger"][0]
    run_id = mlflow_logger.run_id
    # add run ID to model config
    update_model_config_with_run_id(model_name, run_id)


if __name__ == "__main__":
    fire.Fire(main)
