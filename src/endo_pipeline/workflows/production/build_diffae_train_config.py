from typing import Literal

from endo_pipeline.settings.image_data import DIFFAE_DEFAULT_CROP_SIZE
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_IMAGE_TYPE_FOR_SEMANTIC_CONDITIONING,
    DEFAULT_NUM_LATENT_DIMENSIONS,
)


def main(
    model_manifest_name: str | None = None,
    run_name: str | None = None,
    crop_size: int = DIFFAE_DEFAULT_CROP_SIZE,
    condition_on: Literal["bf", "cdh5"] = DEFAULT_IMAGE_TYPE_FOR_SEMANTIC_CONDITIONING,
    latent_dim: int = DEFAULT_NUM_LATENT_DIMENSIONS,
    num_workers: int | None = None,
) -> None:
    """
    Build config for training a DiffAE model.

    #diffae #model-training

    This workflow builds model training configs starting with the base model
    training config and overriding with training run-specific configuration.
    These configurations are saved locally, and then be used by the
    `train-diffae` workflow to train models.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe build-diffae-train-config -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe build-diffae-train-config --datasets DATASET_NAME
    ```

    ## Training run naming

    If a model manifest name is not given, it will be automatically constructed
    based on the crop size, conditioning key, and latent dimension size.

    The training run instantiated from this workflow will be saved in the
    corresponding model manifest, with run name either provided by the user or
    automatically generated to be unique (`run_name = "diffae_TIMESTAMP"`).

    If the user provides a run name that already exists in the manifest, a
    unique name will be generated and a warning will be logged.

    ## Conditioning image type

    The model can be conditioned on either brightfield (`bf`) or CDH5
    fluorescence (`cdh5`) image channels. The conditioning channel is set using
    the `condition_on` parameter via overriding the training config. The default
    is brightfield.

    ## Latent dimension size

    The number of latent dimensions for the DiffAE model can be specified with
    the `latent_dim` parameter. The default is `DEFAULT_NUM_LATENT_DIMENSIONS`
    from `endo_pipeline.settings.workflow_defaults.`

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will set up the
    training config with reduced epochs and modified cache and replaces rates.

    Parameters
    ----------
    model_manifest_name
        An optional name for the model manifest.
    run_name
        An optional name for the training run.
    crop_size
        The length of the 2D image crop in pixels to use for model training.
    condition_on
        The abbreviated name of the image channel to condition the model on.
    latent_dim
        The number of latent dimensions for the DiffAE model.
    num_workers
        Number of workers to use for loading data. If not given, estimate based
        on total number of logical CPUs in the system.
    """

    import logging

    from cyto_dl.api import CytoDLModel

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
    from endo_pipeline.configs import load_model_config
    from endo_pipeline.io import get_output_path, make_name_unique, resolve_dataframe_location
    from endo_pipeline.library.model import get_dataset_names_used_for_training
    from endo_pipeline.library.model.config_overrides import ModelConfigOverrideTrain
    from endo_pipeline.manifests import (
        ModelLocation,
        create_model_manifest,
        load_dataframe_manifest,
        save_model_manifest,
    )
    from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_TRAIN_CONFIG
    from endo_pipeline.settings.workflow_defaults import (
        DIFFAE_IMAGE_LOADING_KEY_PREFIX,
        DIFFAE_TRAIN_DATAFRAME_MANIFEST_PREFIX,
    )

    logger = logging.getLogger(__name__)

    # Adjust workflow settings for demo mode. Append a suffix to the manifest
    # name to indicate that they were generated from a workflow demo, and reduce
    # the number of epochs and logging steps. Also adjust the cache and replace
    # rates. Note that while 100% of the data is used for demo mode, the cache
    # rate for actual training can be adjusted if needed.
    if DEMO_MODE:
        name_suffix = "_demo"
        min_num_epochs = 1
        max_num_epochs = 3
        epoch_multiplier = False
        log_every_n_steps = 1
        cache_rate = 1.0
        replace_rate = 0.1
    else:
        name_suffix = ""
        min_num_epochs = 5000
        max_num_epochs = 20000
        epoch_multiplier = True
        log_every_n_steps = 50
        cache_rate = 1.0
        replace_rate = 0.5

    # Create name components from input parameters
    patch_name = f"_patch_{crop_size}x{crop_size}"
    condition_name = f"_condition_on_{condition_on}"
    latent_name = f"_latent_{latent_dim}"

    # Build dataframe manifest name to load training and validation dataframes.
    # Note that the dataframe manifest name does not include the patch size or
    # conditioning type, as these are not relevant for the dataframe itself.
    dataframe_manifest_name = f"{DIFFAE_TRAIN_DATAFRAME_MANIFEST_PREFIX}{name_suffix}"

    try:
        dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    except FileNotFoundError:
        logger.error(
            "Dataframe manifest '%s' not found. "
            "Please run the create_diffae_train_dataframe workflow first.",
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
    template_config = load_model_config(DIFFAE_MODEL_TRAIN_CONFIG)

    # Build the model manifest name, if not provided.
    if model_manifest_name is None:
        model_manifest_name = f"diffae{patch_name}{condition_name}{latent_name}"
        model_manifest_name = f"{model_manifest_name}{name_suffix}"

    # Create or load the model manifest.
    manifest = create_model_manifest(model_manifest_name, __file__)

    # Build the run name, if not provided.
    if run_name is None:
        run_name = make_name_unique("diffae").name
    elif run_name in manifest.locations:
        run_name = make_name_unique(run_name).name
        logger.warning("Run name already exists in manifest, changed to [ %s ]", run_name)

    logger.info("Model manifest name: [ %s ]", model_manifest_name)
    logger.info("Run name: [ %s ]", run_name)

    # Create training config path.
    config_path = get_output_path(
        "models", model_manifest_name, run_name, "configs", include_timestamp=False
    )
    config_file = config_path / f"train{name_suffix}.yaml"

    # Build the training config overrides.
    overrides = ModelConfigOverrideTrain(
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        crop_size=crop_size,
        condition_key=f"{DIFFAE_IMAGE_LOADING_KEY_PREFIX}{condition_on}",
        latent_dim=latent_dim,
        train_dataframe_path=train_dataframe_path,
        val_dataframe_path=val_dataframe_path,
        min_epochs=min_num_epochs,
        max_epochs=max_num_epochs,
        epoch_multiplier=epoch_multiplier,
        cache_rate=cache_rate,
        replace_rate=replace_rate,
        log_steps=log_every_n_steps,
        num_gpus=NUM_GPUS,
        num_workers=num_workers,
    )

    # Initialize the model with training template and overrides and save config.
    cytodl_model = CytoDLModel()
    cytodl_model.load_config_from_dict(template_config)
    cytodl_model.override_config(overrides.to_dict())
    cytodl_model.save_config(config_file)
    logger.info("Training config saved to [ %s ]", config_file)

    # Get list of training datasets based on dataframes.
    list_of_training_datasets = get_dataset_names_used_for_training(
        train_dataframe_location,
        val_dataframe_location,
    )

    # Populate manifest with training run location and parameters.
    manifest.parameters = {
        "training_datasets": list_of_training_datasets,
        "crop_size": crop_size,
        "condition_on": condition_on,
        "latent_dim": latent_dim,
    }
    manifest.locations[run_name] = ModelLocation()
    save_model_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
