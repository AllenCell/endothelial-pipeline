from pathlib import Path

TAGS = ["production", "diffae", "model_training"]


def main(
    crop_size: int = 128,
    train_csv_path: Path | str | None = None,
    val_csv_path: Path | str | None = None,
) -> None:
    """
    Train a DiffAE model using the provided configuration.

    Parameters
    ----------
    crop_size
        The pixel size of the image crop to use for training. Default is 128.
        This is the crop size along one dimension, the image will be square
        with size (crop_size, crop_size).

    train_csv_path
        The path to the training dataset CSV file. Default is None.
        If None, the default path for the output of `generate_csv_for_training_diffae`
        will be used.

    val_csv_path
        The path to the validation dataset CSV file. Default is None.
        If None, the default pathfor the output of `generate_csv_for_training_diffae`
        will be used.

    Returns
    -------
    None
        The function creates and saves a ModelConfig object with the trained model's
        MLflow run ID and the list of datasets used for training.
    """
    import datetime

    from omegaconf import OmegaConf

    from src.endo_pipeline.configs import ModelConfig, save_model_config
    from src.endo_pipeline.library.model import (
        get_dataset_names_used_for_training,
        get_model_dir,
        get_valid_csv_path_for_training,
        initialize_diffae_model,
    )

    # get valid CSV paths for training and validation datasets
    train_csv_path = get_valid_csv_path_for_training(train_csv_path, "train")
    val_csv_path = get_valid_csv_path_for_training(val_csv_path, "val")

    # load training config
    template_training_config = OmegaConf.load(get_model_dir() / "diffae_training.yaml")

    # set model name via timestamp and crop size
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"diffae_patch_{crop_size}x{crop_size}_{timestamp}"

    # initialize DiffAE model: generates config
    # overrides and sets up output directories
    model = initialize_diffae_model(
        template_training_config,
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
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
