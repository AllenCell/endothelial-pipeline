from typing import Literal

TAGS = ["diffae_model_finetuning"]


def main(
    model_name: str = "diffae_04_10",
    dataset_pair_type: Literal["live_fixed", "20x_40x"] = "live_fixed",
    resolution_level: int = 1,
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
    resolution_level
        The resolution level of the zarr files to be used for training.

    Returns
    -------
    :
        The function creates and save a :code:`ModelConfig` object with the finetuned model's
        MLflow run ID and the list of datasets used for training.
    """
    import datetime
    import logging
    from typing import cast

    from omegaconf import OmegaConf

    from src.endo_pipeline import TESTING_MODE
    from src.endo_pipeline.configs import CytoDLModelConfig, load_model_config, save_model_config
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.model import (
        download_mlflow_artifact,
        get_ckpt_path,
        get_dataset_names_used_for_training,
        get_model_dir,
        get_valid_dataframe_path_for_training,
        initialize_diffae_model_for_finetuning,
    )
    from src.endo_pipeline.manifests import load_dataframe_manifest

    logger = logging.getLogger(__name__)

    # get training and validation datasets based on zarr resolution
    # by loading the DataframeManifest from the model directory
    # and using the DatasetLocation objects to get the paths
    manifest_name = f"diffae_finetuning_dataframe_resolution_{resolution_level}"
    if TESTING_MODE:
        manifest_name += "_test_workflow"
    try:
        dataframe_manifest = load_dataframe_manifest(manifest_name)
    except FileNotFoundError:
        logger.error(
            "Dataframe manifest [ %s ] for resolution_level [ %s ] not found. "
            "Please run the create_diffae_finetuning_dataframe script first "
            "with the appropriate resolution_level.",
            manifest_name,
            resolution_level,
        )
        raise
    train_dataframe_location = dataframe_manifest.locations["training"]
    val_dataframe_location = dataframe_manifest.locations["validation"]

    # get paths from the DataframeLocation objects
    # to pass into the DiffAE model training script
    # (need for training config setup and CytoDL dataloaders)
    train_dataframe_path = get_valid_dataframe_path_for_training(train_dataframe_location)
    val_dataframe_path = get_valid_dataframe_path_for_training(val_dataframe_location)

    model_save_path = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
    )

    # download model to finetune
    model_config = cast(CytoDLModelConfig, load_model_config(model_name))
    finetune_run_id = model_config.mlflow_run_id
    diffae_ckpt_path = get_ckpt_path(run_id=finetune_run_id)
    download_mlflow_artifact(finetune_run_id, diffae_ckpt_path, model_save_path)

    # get template config
    template_finetune_config = OmegaConf.load(get_model_dir() / "diffae_finetune.yaml")

    # initialize model for finetuning
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
    finetuned_model_name = f"{model_name}_finetuned_for_{dataset_pair_type}_{timestamp}"
    model = initialize_diffae_model_for_finetuning(
        template_finetune_config=template_finetune_config,
        finetuned_model_name=finetuned_model_name,
        train_dataframe_path=train_dataframe_path,
        val_dataframe_path=val_dataframe_path,
        model_save_path=model_save_path,
        diffae_ckpt_path=diffae_ckpt_path,
        max_num_epochs=100 if not TESTING_MODE else 1,
        log_every_n_steps=50 if not TESTING_MODE else 1,
    )
    # save the model config locally instead of printing
    local_config_save_path = get_output_path("models", "training_configs")
    model.save_config(local_config_save_path / f"{finetuned_model_name}_train.yaml")
    # call train method to start finetuning
    _, object_dict = model.train()

    # retrive MLflow run ID
    mlflow_logger = object_dict["logger"][0]
    run_id = mlflow_logger.run_id
    # get list of datasets used for training
    # THIS NEEDS TO BE REFACTORED TO USE THE DATAFRAME MANIFEST
    list_of_training_datasets = get_dataset_names_used_for_training(
        train_dataframe_location, val_dataframe_location, f"{dataset_pair_type}_paired_datasets"
    )
    # add run ID and training datasets to model config
    model_config = CytoDLModelConfig(
        name=finetuned_model_name,
        mlflow_run_id=run_id,
        training_datasets=list_of_training_datasets,
    )
    # save the model config
    save_model_config(model_config)


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
