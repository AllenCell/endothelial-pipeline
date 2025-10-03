from typing import Literal

TAGS = ["diffae_model_finetuning"]


def main(
    base_model_manifest_name: str = "diffae_04_10",
    base_run_name: str | None = None,
    finetuned_model_manifest_name: str | None = None,
    finetuned_run_name: str | None = None,
    dataset_pair_type: Literal["live_fixed", "20x_40x"] = "live_fixed",
    resolution_level: int = 1,
) -> None:
    """
    Finetune a DiffAE model to align features for paired datasets.

    **Training run naming**

    If a finetuned model manifest name is not given, it will be automatically constructed based
    on the dataset pair type and the resolution level of the zarr files. The format is
    ``finetune_{dataset_pair_type}_resolution_{resolution_level}``.

    The training runs instantiated from the this workflow will be saved in the
    corresponding model manifest, with run name either provided by the user or automatically
    generated to be unique ( ``run_name = f"diffae_{timestamp}"`` ).

    If the user provides a run name that already exists in the manifest, a unique name will be
    generated and a warning will be logged.

    Parameters
    ----------
    base_model_manifest_name
        Name of the model manifest to load the baseline model from.
    base_run_name
        Name of the baseline model run to apply. If None, uses the most recent run.
    finetuned_model_manifest_name
        Optional, name of the model manifest to save the finetuned model to.
    finetuned_run_name
        Optional, name to give the finetuned model run.
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
    import logging
    from pathlib import Path

    from endo_pipeline import DEMO_MODE, NUM_GPUS
    from endo_pipeline.io import (
        get_output_path,
        load_model,
        load_model_config_from_path,
        make_name_unique,
        resolve_dataframe_location,
    )
    from endo_pipeline.library.model import (
        get_dataset_names_used_for_training,
        initialize_diffae_model_for_finetuning,
    )
    from endo_pipeline.manifests import (
        ModelLocation,
        ModelManifest,
        get_model_location_for_run,
        load_dataframe_manifest,
        load_model_manifest,
        save_model_manifest,
    )
    from endo_pipeline.settings import RELATIVE_PATH_TO_FINETUNE_CONFIG

    logger = logging.getLogger(__name__)

    # Adjust workflow settings for demo mode. Append a suffix to the manifest
    # and model names to indicate that they were generated from a workflow demo,
    # and reduce the number of epochs and logging steps.
    if DEMO_MODE:
        name_suffix = "_demo"
        max_num_epochs = 1
        log_every_n_steps = 1
        cache_rate = 1.0
        replace_rate = 0.1
    else:
        name_suffix = ""
        max_num_epochs = 100
        log_every_n_steps = 50
        cache_rate = 1.0  # This can be changed!
        replace_rate = 0.5

    # double-check zarr resolution from baseline model manifest parameters
    base_model_manifest = load_model_manifest(base_model_manifest_name)
    if "resolution_level" in base_model_manifest.parameters:
        baseline_resolution = base_model_manifest.parameters["resolution_level"]
        if baseline_resolution != resolution_level:
            logger.warning(
                "Baseline model [ %s ] was trained at resolution level [ %s ], "
                "but finetuning is being run at resolution level [ %s ]. "
                "This may lead to suboptimal results.",
                base_model_manifest_name,
                baseline_resolution,
                resolution_level,
            )

    # get training and validation datasets based on zarr resolution
    # by loading the DataframeManifest from the model directory
    # and using the DatasetLocation objects to get the paths
    dataframe_manifest_base = "diffae_finetuning_dataframe_resolution_"
    dataframe_manifest_name = f"{dataframe_manifest_base}{resolution_level}{name_suffix}"

    try:
        dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    except FileNotFoundError:
        logger.error(
            "Dataframe manifest [ %s ] for resolution_level [ %s ] not found. "
            "Please run the create_diffae_finetuning_dataframe script first "
            "with the appropriate resolution_level.",
            dataframe_manifest_name,
            resolution_level,
        )
        raise

    train_dataframe_location = dataframe_manifest.locations["training"]
    val_dataframe_location = dataframe_manifest.locations["validation"]

    # get paths from the DataframeLocation objects
    # to pass into the DiffAE model training script
    # (need for training config setup and CytoDL dataloaders)
    train_dataframe_path = resolve_dataframe_location(train_dataframe_location)
    val_dataframe_path = resolve_dataframe_location(val_dataframe_location)

    # get template config
    template_finetune_config = load_model_config_from_path(RELATIVE_PATH_TO_FINETUNE_CONFIG)

    # initialize baseline model for finetuning
    base_model_manifest = load_model_manifest(base_model_manifest_name)
    base_run_name = (
        list(base_model_manifest.locations.keys())[-1] if base_run_name is None else base_run_name
    )
    base_model_location = get_model_location_for_run(base_model_manifest, base_run_name)
    base_model = load_model(base_model_location)

    # set up manifest and run names for finetuned model
    if finetuned_model_manifest_name is None:
        finetuned_model_manifest_name = (
            f"finetune_{dataset_pair_type}_resolution_" f"{resolution_level}{name_suffix}"
        )
    if finetuned_run_name is None:
        # Default is "finetuned_diffae_{timestamp}"
        finetuned_run_name = make_name_unique("finetuned_diffae").name
    else:
        # If run name provided, make sure it's unique within the manifest
        if finetuned_run_name in load_model_manifest(finetuned_model_manifest_name).locations:
            # If it's not unique, make it so and log a warning
            finetuned_run_name = make_name_unique(finetuned_run_name).name
            logger.warning(
                "Provided run name already exists in manifest, changed current run name to [ %s ]",
                finetuned_run_name,
            )
    logger.info("Finetuned model manifest name: [ %s ]", finetuned_model_manifest_name)
    logger.info("Finetuned run name: [ %s ]", finetuned_run_name)

    model = initialize_diffae_model_for_finetuning(
        base_model=base_model,
        template_finetune_config=template_finetune_config,
        finetuned_model_manifest_name=finetuned_model_manifest_name,
        finetuned_run_name=finetuned_run_name,
        train_dataframe_path=train_dataframe_path,
        val_dataframe_path=val_dataframe_path,
        max_num_epochs=max_num_epochs,
        log_every_n_steps=log_every_n_steps,
        cache_rate=cache_rate,
        replace_rate=replace_rate,
        num_gpus=NUM_GPUS,
    )
    # save the input model config locally instead of printing
    local_config_save_path = get_output_path(
        "models", "training_configs", finetuned_model_manifest_name, finetuned_run_name
    )
    model_config_save_path = local_config_save_path / "train.yaml"
    model.save_config(model_config_save_path)
    logger.info(
        "Training config saved to [ %s ]",
        local_config_save_path / "train.yaml",
    )
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

    # Create a new model manifest with workflow parameters, if a matching
    # manifest does not already exist. Add the model training run to the list
    # of manifest locations.
    try:
        manifest = load_model_manifest(finetuned_model_manifest_name)
    except FileNotFoundError:
        logger.info(
            "Model manifest [ %s ] not found, creating a new one.",
            finetuned_model_manifest_name,
        )
        parameters = {
            "training_datasets": list_of_training_datasets,
            "resolution_level": resolution_level,
        }
        manifest = ModelManifest(
            name=finetuned_model_manifest_name, parameters=parameters, workflow=Path(__file__).stem
        )

    manifest.locations[finetuned_run_name] = ModelLocation(mlflowid=run_id)
    save_model_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
