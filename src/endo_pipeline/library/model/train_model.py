import os
from pathlib import Path
from typing import Literal

from cyto_dl.api import CytoDLModel
from omegaconf import DictConfig, ListConfig

from src.endo_pipeline.io import get_output_path


def get_model_dir() -> Path:
    """Get the path to `src.endo_pipeline.library.model`."""
    return Path(__file__).resolve().parent


def _generate_overrides_for_model_training(
    model_name: str,
    crop_size: int,
    train_csv_path: Path,
    val_csv_path: Path,
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


def _generate_overrides_for_finetuning(
    model_name: str,
    dataset_pair_type: Literal["live_fixed", "20x_40x"],
    train_csv_path: Path,
    val_csv_path: Path,
    ckpt_path: Path,
) -> dict:
    """
    Generate overrides for finetuning a DiffAE model.

    Parameters
    ----------
    model_name: str
        The name of the model to finetune. This should correspond to a
        directory in `results/models/` and match the model name used during the
        `paired_data_validation` step.
    dataset_pair_type: Literal['live_fixed', '20x_40x']
        The type of dataset to use for finetuning. This should match the dataset
        type used during the `paired_data_validation` step.
    train_csv_path: Path
        The path to the training CSV file containing paired data.
    val_csv_path: Path
        The path to the validation CSV file containing paired data.
    ckpt_path: Path
        The path to the DiffAE checkpoint to finetune.
    """
    # create output directories if they do not exist
    save_path = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        include_timestamp=False,
    )
    _ = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        "checkpoints",
        include_timestamp=False,
    )
    _ = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        "logs",
        include_timestamp=False,
    )

    overrides = {
        # point to already projected paired dataset
        "data.train_dataloaders.dataset.csv_path": str(train_csv_path),
        "data.val_dataloaders.dataset.csv_path": str(val_csv_path),
        # load diffae checkpoint to finetune
        "checkpoint.ckpt_path": str(ckpt_path),
        "checkpoint.weights_only": True,
        "checkpoint.strict": False,
        # save to user-specified directory
        "model.save_dir": (save_path / "logs").as_posix(),
        "trainer.default_root_dir": save_path,
        "callbacks.model_checkpoint.dirpath": (save_path / "checkpoints").as_posix(),
        "paths.output_dir": (save_path / "logs").as_posix(),
        # do training
        "train": True,
        # # make sure that last ckpt is saved
        # "callbacks.model_checkpoint.monitor": None,
        # updated mlflow logger
        "logger": {
            "mlflow": {
                "_target_": "cyto_dl.loggers.MLFlowLogger",
                "tracking_uri": "https://production.int.allencell.org/mlflow/",
                "experiment_name": "endo_diffae",
                "run_name": "fixed_finetune_separate_encoder",
            }
        },
    }

    return overrides


def initialize_diffae_model(
    template_training_config: DictConfig | ListConfig,
    crop_size: int,
    model_name: str,
    train_csv_path: Path,
    val_csv_path: Path,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    Parameters
    ----------
    template_training_config
        The template training configuration to use.
    crop_size
        The pixel size of the image crop to use for training.
        This is the crop size along one dimension, the image will be square
        with size (crop_size, crop_size).
    model_name
        The name of the model to train.
    train_csv_path
        The path to the training dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.
    val_csv_path
        The path to the validation dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.

    Returns
    -------
    cytodl_model
        An initialized CytoDLModel for training the DiffAE model.
    """
    # user overrides for training
    overrides = _generate_overrides_for_model_training(
        model_name, crop_size, train_csv_path, val_csv_path
    )

    # init model
    cytodl_model = CytoDLModel()
    # override config with workflow inputs
    cytodl_model.load_config_from_dict(template_training_config)
    cytodl_model.override_config(overrides)
    return cytodl_model


def get_valid_csv_path_for_training(
    csv_path: Path | str | None, csv_name: Literal["train", "val"]
) -> Path:
    """
    Get a valid CSV path for training or validation datasets.

    Parameters
    ----------
    csv_path
        The path to the CSV file, defaults to None. If None, the default path
        for the output of `generate_csv_for_training_diffae` will be used.
    csv_name
        The name of the CSV file to validate, "train" or "val".
        If csv_path is not None, csv_name will not be used in the path generation.
        This input is used for the default case where csv_path is None,
        and the path will be generated based on the csv_name (train or val).


    Returns
    -------
    csv_path
        A valid Path object pointing to the CSV file.
    """
    if csv_path is None:
        csv_path = get_output_path("manifests", include_timestamp=False) / f"{csv_name}.csv"

    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please provide a valid path.")

    return csv_path


def initialize_diffae_model_for_finetuning(
    template_finetune_config: DictConfig | ListConfig,
    model_name: str,
    dataset_pair_type: Literal["live_fixed", "20x_40x"],
    train_csv_path: Path,
    val_csv_path: Path,
    model_save_path: Path,
    diffae_ckpt_path: Path,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    Parameters
    ----------
    template_finetune_config
        The template configuration for finetuning the DiffAE model.
    model_name
        The name of the model to train.
    dataset_pair_type
        The type of dataset to use for finetuning ("live_fixed" or "20X_40X").
        This should match the input used during the `paired_data_validation` step.
    train_csv_path
        The path to the training dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.
    val_csv_path
        The path to the validation dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.
    model_save_path
        The path to the directory where the checkpoints and logs will be saved.
    diffae_ckpt_path
        The path to the DiffAE checkpoint to finetune. This should be a path
        to the checkpoint downloaded from MLflow artifacts.

    Returns
    -------
    cytodl_model
        An initialized CytoDLModel for finetuning the DiffAE model.
    """
    # generate overrides for train.yaml for finetuning
    overrides = _generate_overrides_for_finetuning(
        model_name=model_name,
        dataset_pair_type=dataset_pair_type,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        ckpt_path=model_save_path / diffae_ckpt_path,
    )

    # init model
    cytodl_model = CytoDLModel()
    cytodl_model.load_config_from_dict(template_finetune_config)
    cytodl_model.override_config(overrides)

    return cytodl_model


def get_valid_csv_path_for_finetuning(
    csv_path: Path | str | None,
    csv_name: Literal["train", "val"],
    dataset_pair_type: Literal["live_fixed", "20x_40x"],
) -> Path:
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
    dataset_pair_type: Literal["live_fixed", "20x_40x"]
        The type of dataset to use for finetuning. This should match the dataset
        type used during the `paired_data_validation` step.


    Returns
    -------
    Path
        A valid Path object pointing to the CSV file.
    """
    if csv_path is None:
        csv_path = (
            get_output_path("finetune_paired_dataset", dataset_pair_type, include_timestamp=False)
            / f"{csv_name}.csv"
        )

    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please provide a valid path.")

    return csv_path
