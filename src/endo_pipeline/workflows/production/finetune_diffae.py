from pathlib import Path
from typing import Literal

TAGS = ["diffae_model_finetuning"]


def main(
    model_name: str = "diffae_04_10",
    dataset_pair_type: Literal["live_fixed", "20x_40x"] = "live_fixed",
    train_csv_path: Path | None = None,
    val_csv_path: Path | None = None,
) -> None:
    """
    Finetune a DiffAE model to align features for paired datasets.

    Parameters
    ----------
    model_name
        The name of the model to use for finetuning (should correspond to a
        config in :code:`results/models/`).
    dataset_pair_type
        The type of dataset pairs to use for finetuning ("live_fixed" or "20x_40x").
    train_csv_path
        Optional user-specified path to the training dataset CSV file.
    val_csv_path
        Optional user-specified path to the validation dataset CSV file.

    Returns
    -------
    :
        The function creates and save a :code:`ModelConfig` object with the finetuned model's
        MLflow run ID and the list of datasets used for training.
    """
    from typing import cast

    from omegaconf import OmegaConf

    from endo_pipeline.configs import CytoDLModelConfig, load_model_config, save_model_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model import (
        download_mlflow_artifact,
        get_ckpt_path,
        get_dataset_names_used_for_training,
        get_model_dir,
        get_valid_csv_path_for_finetuning,
        initialize_diffae_model_for_finetuning,
    )

    # get valid CSV paths for training and validation datasets
    train_csv_path = get_valid_csv_path_for_finetuning(train_csv_path, "train", dataset_pair_type)
    val_csv_path = get_valid_csv_path_for_finetuning(val_csv_path, "val", dataset_pair_type)

    model_save_path = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        include_timestamp=False,
    )

    # download model to finetune
    model_config = cast(CytoDLModelConfig, load_model_config(model_name))
    finetune_run_id = model_config.mlflow_run_id
    diffae_ckpt_path = get_ckpt_path(run_id=finetune_run_id)
    download_mlflow_artifact(finetune_run_id, diffae_ckpt_path, model_save_path)

    # get template config
    template_finetune_config = OmegaConf.load(get_model_dir() / "diffae_finetune.yaml")

    # initialize model for finetuning
    model = initialize_diffae_model_for_finetuning(
        template_finetune_config=template_finetune_config,
        model_name=model_name,
        dataset_pair_type=dataset_pair_type,
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
    # THIS NEEDS TO BE REFACTORED TO USE THE DATAFRAME MANIFEST
    list_of_training_datasets = get_dataset_names_used_for_training(
        train_csv_path, val_csv_path, f"{dataset_pair_type}_paired_datasets"
    )
    # add run ID and training datasets to model config
    model_config = CytoDLModelConfig(
        name=f"{model_name}_finetuned_for_{dataset_pair_type}",
        mlflow_run_id=run_id,
        training_datasets=list_of_training_datasets,
    )
    # save the model config
    save_model_config(model_config)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
