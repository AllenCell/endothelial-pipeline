from pathlib import Path
from typing import Any, Literal

import fire
from cyto_dl.api import CytoDLModel

from src.endo_pipeline.configs import ModelConfig, load_model_config, save_model_config
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.model import (
    download_mlflow_artifact,
    generate_overrides_for_finetuning,
    get_ckpt_path,
    get_dataset_names_used_for_training,
)


def _initialize_diffae_model_for_finetuning(
    model_name: str,
    dataset_type: Literal["live_fixed", "20x_40x"],
    train_csv_path: Path,
    val_csv_path: Path,
    model_save_path: Path,
    diffae_ckpt_path: Path,
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
    model_save_path: Path
        The path to the directory where the checkpoints and logs will be saved.
    diffae_ckpt_path: Path
        The path to the DiffAE checkpoint to finetune. This should be a path
        to the checkpoint downloaded from MLflow artifacts.
    """
    # generate overrides for train.yaml for finetuning
    overrides = generate_overrides_for_finetuning(
        model_name=model_name,
        dataset_type=dataset_type,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        ckpt_path=model_save_path / diffae_ckpt_path,
    )

    # init model
    model = CytoDLModel()
    model.load_config_from_file(model_save_path / "config" / "train.yaml")
    model.override_config(overrides)
    model.train()

    return model


def _get_valid_csv_path(
    csv_path: Path | str | None,
    csv_name: Literal["train", "val"],
    dataset_type: Literal["live_fixed", "20x_40x"],
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


    Returns
    -------
    Path
        A valid Path object pointing to the CSV file.
    """
    if csv_path is None:
        csv_path = (
            get_output_path("finetune_paired_dataset", dataset_type, include_timestamp=False)
            / f"{csv_name}.csv"
        )

    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please provide a valid path.")

    return csv_path


def main(
    model_name: str = "diffae_04_10",
    dataset_type: Literal["live_fixed", "20x_40x"] = "live_fixed",
    model_template_name: str = "live_fixed_finetune_template",
    train_csv_path: Path | None = None,
    val_csv_path: Path | None = None,
):
    """
    Finetune a DiffAE model with paired live/fixed data.

    Parameters
    ----------
    model_name: str
        The name of the model to use for finetuning. This should correspond to a
        directory in `results/models/` and match the model name used during the
        `paired_data_validation` step.
    dataset_type: Literal['live_fixed', '20x_40x']
        The type of dataset to use for finetuning. This should match the dataset
        type used during the `paired_data_validation` step.
    model_template_name: str
        The name of the model template to use for finetuning.
        This should correspond to a model in `src/endo_pipeline/configs/model`
    train_csv_path: Path | None
        The path to the training CSV file containing paired data.
        If None, the default path for the output of paired_fixed_live_validation will be used.
    val_csv_path: Path | None
        The path to the validation CSV file containing paired data.
        If None, the default path for the output of paired_fixed_live_validation will be used.
    """

    # get valid CSV paths for training and validation datasets
    train_csv_path = _get_valid_csv_path(train_csv_path, "train")
    val_csv_path = _get_valid_csv_path(val_csv_path, "val")

    model_save_path = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_type}",
        include_timestamp=False,
    )

    # download model to finetune
    finetune_run_id = load_model_config(model_name).mlflow_run_id
    diffae_ckpt_path = get_ckpt_path(run_id=finetune_run_id)
    download_mlflow_artifact(finetune_run_id, diffae_ckpt_path, model_save_path)

    # get template config
    template_run_id = load_model_config(model_template_name).mlflow_run_id
    download_mlflow_artifact(template_run_id, "config/train.yaml", model_save_path)

    # initialize model for finetuning
    model = _initialize_diffae_model_for_finetuning(
        model_name=model_name,
        dataset_type=dataset_type,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        model_save_path=model_save_path,
        diffae_ckpt_path=diffae_ckpt_path,
    )
    # call train method to start finetuning
    _, object_dict = model.train()

    # retrive MLflow run ID
    mlflow_logger = object_dict["logger"][0]
    run_id = mlflow_logger.run_id
    # get list of datasets used for training
    list_of_training_datasets = get_dataset_names_used_for_training(
        train_csv_path, val_csv_path, "paired_datasets"
    )
    # add run ID and training datasets to model config
    model_config = ModelConfig(
        name=f"finetune_{model_name}_on_{dataset_type}",
        mlflow_run_id=run_id,
        training_datasets=list_of_training_datasets,
    )
    # save the model config
    save_model_config(model_config)


if __name__ == "__main__":
    fire.Fire(main)
