from pathlib import Path

TAGS = ["diffae_model_training"]


def main(
    zarr_resolution: int = 1,
    crop_size: int = 128,
    train_csv_path: Path | str | None = None,
    val_csv_path: Path | str | None = None,
) -> None:
    """
    Train a DiffAE model using the provided configuration.

    Parameters
    ----------
    zarr_resolution
        The resolution level of the zarr files to be used for training. Default is 1,
        which corresponds to downsampling by half.
    crop_size
        The length of the 2D image crop in pixels to use for model training.
    train_csv_path
        Optional user-specified path to the training dataset CSV file.
    val_csv_path
        Optional user-specified path to the validation dataset CSV file.

    Returns
    -------
    :
        The function creates and saves a ModelConfig object with the trained model's
        MLflow run ID and the list of datasets used for training.
    """
    import datetime
    import logging

    from omegaconf import OmegaConf

    from src.endo_pipeline.configs import CytoDLModelConfig, save_model_config
    from src.endo_pipeline.library.model import (
        get_dataset_names_used_for_training,
        get_model_dir,
        get_valid_csv_path_for_training,
        initialize_diffae_model,
    )

    # set lightning logger level to WARNING to avoid excessive logging
    lightning_logger = logging.getLogger("lightning.pytorch")
    lightning_logger.setLevel(logging.WARNING)

    # get valid CSV paths for training and validation datasets based on zarr resolution

    train_dataframe_location = ""
    train_csv_path = get_valid_csv_path_for_training(train_csv_path, zarr_resolution, "train")
    val_csv_path = get_valid_csv_path_for_training(val_csv_path, zarr_resolution, "val")

    # load template training config
    template_training_config = OmegaConf.load(get_model_dir() / "diffae_training.yaml")

    # set model name via zarr resolution, crop size, and current timestamp
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"diffae_resolution_{zarr_resolution}_patch_{crop_size}x{crop_size}_{timestamp}"

    # initialize DiffAE model: generates config overrides and sets up output directories
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
    model_config = CytoDLModelConfig(
        name=model_name,
        mlflow_run_id=run_id,
        training_datasets=list_of_training_datasets,
    )
    # save the model config
    save_model_config(model_config)


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
