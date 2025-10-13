from typing import Annotated

from cyclopts import Parameter

TAGS = ["diffae", "model_training"]


def main(
    model_manifest_name: str | None = None,
    run_name: str | None = None,
    resolution_level: int = 1,
    crop_size: int = 128,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = True,
) -> None:
    """
    Build config for training a DiffAE model.

    **Training run naming**

    If a model manifest name is not given, it will be automatically constructed
    based on the resolution level of the zarr files, the crop size, and whether
    cell piling exclusion is enabled or not.

    The training run instantiated from this workflow will be saved in the
    corresponding model manifest, with run name either provided by the user or
    automatically generated to be unique (``run_name = "diffae_TIMESTAMP"``).

    If the user provides a run name that already exists in the manifest, a
    unique name will be generated and a warning will be logged.

    **Cell piling exclusion**

    By default, timepoints with cell piling annotations are included in the
    training and validation datasets from ``create-diffae-training-dataframe``,
    unless ``include_cell_piling`` is False. This means that by default, the
    model will be trained on data that includes cell piling. To train a model
    that does not "see" cell piling,  run ``create-diffae-training-dataframe``
    with the flag ``--exclude-cell-piling`` and then run this training script
    with the same flag.

    When ``include_cell_piling`` is True, the workflow will use the "standard"
    dataframe manifest ``diffae_training_dataframe_resolution_RESOLUTION`` for
    training. When ``include_cell_piling`` is False, the workflow will find the
    dataframe manifest with the suffix ``_exclude_cell_piling`` and use the
    corresponding training and validation datasets.

    **Workflow demo**

    If demo mode is enabled, this workflow will set up the training config with
    reduced epochs and modified cache and replaces rates. The config will have
    the suffix ``_test_workflow``.

    Parameters
    ----------
    model_manifest_name
        An optional name for the model manifest.
    run_name
        An optional name for the training run.
    resolution_level
        The resolution level of the zarr files to be used for training.
    crop_size
        The length of the 2D image crop in pixels to use for model training.
    include_cell_piling
        Include cell piling timepoints if True, exclude them if False.
    """

    import logging

    from cyto_dl.api import CytoDLModel

    from endo_pipeline import DEMO_MODE, NUM_GPUS
    from endo_pipeline.configs import load_model_config
    from endo_pipeline.io import get_output_path, make_name_unique, resolve_dataframe_location
    from endo_pipeline.library.model import get_dataset_names_used_for_training
    from endo_pipeline.library.model.model_config_overrides import ModelConfigOverride
    from endo_pipeline.manifests import (
        ModelLocation,
        create_model_manifest,
        load_dataframe_manifest,
        load_model_manifest,
        save_model_manifest,
    )
    from endo_pipeline.settings import DIFFAE_MODEL_TRAIN_CONFIG

    logger = logging.getLogger(__name__)

    # Adjust workflow settings for demo mode. Append a suffix to the manifest
    # name to indicate that they were generated from a workflow demo, and reduce
    # the number of epochs and logging steps. Also adjust the cache and replace
    # rates. Note that while 100% of the data is used for demo mode, the cache
    # rate for actual training can be adjusted if needed.
    if DEMO_MODE:
        name_suffix = "_test_workflow"
        max_num_epochs = 1
        log_every_n_steps = 1
        cache_rate = 1.0
        replace_rate = 0.1
    else:
        name_suffix = ""
        max_num_epochs = 1000
        log_every_n_steps = 50
        cache_rate = 1.0
        replace_rate = 0.5

    # Create name components from input parameters
    res_name = f"_resolution_{resolution_level}"
    patch_name = f"_patch_{crop_size}x{crop_size}"
    piling_name = "_include_cell_piling" if include_cell_piling else "_exclude_cell_piling"

    # Build dataframe manifest name
    dataframe_manifest_name = f"diffae_training_dataframe{res_name}{piling_name}{name_suffix}"

    try:
        dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    except FileNotFoundError:
        logger.error(
            "Dataframe manifest [ %s ] not found. "
            "Please run the create_diffae_training_dataframe script first "
            "with matching settings for resolution level and cell piling.",
            dataframe_manifest_name,
        )
        raise

    # Get training and validation dataframe locations.
    train_dataframe_location = dataframe_manifest.locations["training"]
    val_dataframe_location = dataframe_manifest.locations["validation"]

    # Resolve training and validation dataframe locations to paths.
    train_dataframe_path = resolve_dataframe_location(train_dataframe_location)
    val_dataframe_path = resolve_dataframe_location(val_dataframe_location)

    # Load template training config.
    template_training_config = load_model_config(DIFFAE_MODEL_TRAIN_CONFIG)

    # Build the model manifest name, if not provided.
    if model_manifest_name is None:
        model_manifest_name = f"diffae{res_name}{patch_name}{piling_name}"

    # Build the run name, if not provided.
    if run_name is None:
        run_name = make_name_unique("diffae").name
    elif run_name in load_model_manifest(model_manifest_name).locations:
        run_name = make_name_unique(run_name).name
        logger.warning("Run name already exists in manifest, changed to [ %s ]", run_name)

    logger.info("Model manifest name: [ %s ]", model_manifest_name)
    logger.info("Run name: [ %s ]", run_name)

    # Create training config path.
    config_path = get_output_path(
        "models", model_manifest_name, run_name, "configs", include_timestamp=False
    )
    config_file = config_path / "train.yaml"

    # Build the training config overrides.
    overrides = ModelConfigOverride(
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        task_name="train",
        crop_size=crop_size,
        train_dataframe_path=train_dataframe_path,
        val_dataframe_path=val_dataframe_path,
        max_epochs=max_num_epochs,
        cache_rate=cache_rate,
        replace_rate=replace_rate,
        log_steps=log_every_n_steps,
        num_gpus=NUM_GPUS,
    )

    # Initialize the model with training template and overrides and save config.
    cytodl_model = CytoDLModel()
    cytodl_model.load_config_from_dict(template_training_config)
    cytodl_model.override_config(overrides.to_dict())
    cytodl_model.save_config(config_file)
    logger.info("Training config saved to [ %s ]", config_file)

    # Get list of training datasets based on dataframes.
    list_of_training_datasets = get_dataset_names_used_for_training(
        train_dataframe_location,
        val_dataframe_location,
        "live_20X_objective_3i_microscope",
    )

    # Create a new model manifest with workflow parameters, if a matching
    # manifest does not already exist. Add the model training run to the list
    # of manifest locations.
    manifest = create_model_manifest(model_manifest_name, __file__)
    manifest.parameters = {
        "training_datasets": list_of_training_datasets,
        "crop_size": crop_size,
        "resolution_level": resolution_level,
        "include_cell_piling": include_cell_piling,
    }
    manifest.locations[run_name] = ModelLocation()
    save_model_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
