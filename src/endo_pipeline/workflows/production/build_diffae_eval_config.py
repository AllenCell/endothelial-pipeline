from endo_pipeline.cli import Datasets, PatchType
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    patch_type: PatchType,
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    config_name: str | None = None,
    num_workers: int | None = None,
) -> None:
    """
    Build config for evaluating a DiffAE model.

    #diffae #model-evaluation #gpu

    This workflow builds model evaluation configs for each dataset starting with
    the config used to train the model, then overriding with evaluation-specific
    configuration. These configurations are saved locally, and can then be used
    by the `eval-diffae` workflow to calculate latent features.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe build-diffae-eval-config PATCH_TYPE -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe build-diffae-eval-config PATCH_TYPE --datasets DATASET_NAME
    ```

    ## Patch types

    Two patch types are supported for model evaluation: `grid-based` or
    `cell-centered`. Specific configuration for model evaluation on grid-based
    vs. cell-centered patches are applied to the base model configuration.

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will set up the
    evaluation config for a single dataset.

    Parameters
    ----------
    patch_type
        Patch type used for model evaluation.
    datasets
        List of datasets or dataset collections.
    model_manifest_name
        Name for the model manifest to use for evaluation.
    run_name
        Name for the model run to use for evaluation.
    config_name
        Evaluation override config applied over the trained model config.
    num_workers
        Number of workers to use for loading data. If not given, estimate based
        on total number of logical CPUs in the system.
    """

    import logging

    from cyto_dl.api import CytoDLModel

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection, load_model_config
    from endo_pipeline.io import get_output_path, resolve_dataframe_location
    from endo_pipeline.library.model.config_overrides import ModelConfigOverrideEval
    from endo_pipeline.library.model.eval_model import load_model_for_inference
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        get_dataframe_location_for_dataset,
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_EVAL_CONFIG
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
        DIFFAE_EVAL_DATAFRAME_MANIFEST_PREFIX,
    )

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # When running workflow in demo mode, only evaluate the first dataset.
    if DEMO_MODE:
        logger.warning("DEMO MODE - Only evaluating first dataset")
        datasets = datasets[:1]

    # Build dataframe manifest name to load evaluation dataframes.
    name_suffix = "_demo" if DEMO_MODE else ""
    dataframe_manifest_name = f"{DIFFAE_EVAL_DATAFRAME_MANIFEST_PREFIX}_{patch_type}{name_suffix}"

    try:
        dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    except FileNotFoundError:
        logger.error(
            "Dataframe manifest '%s' not found. "
            "Please run the create_diffae_eval_dataframe workflow first "
            "with matching settings for patch type.",
            dataframe_manifest_name,
        )
        raise

    # Load model manifest and get run name.
    model_manifest = load_model_manifest(model_manifest_name)
    run_name = get_most_recent_run_name(model_manifest) if run_name is None else run_name

    # Load model based on given model manifest, run name, and eval override config.
    model_config_name = DIFFAE_MODEL_EVAL_CONFIG if config_name is None else config_name
    eval_config = load_model_config(model_config_name)
    base_config = load_model_for_inference(model_manifest, run_name, eval_config).cfg

    logger.info("Model manifest name: [ %s ]", model_manifest_name)
    logger.info("Run name: [ %s ]", run_name)

    # Create or load the feature dataframe manifest.
    feature_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, patch_type
    )
    feature_manifest = create_dataframe_manifest(f"{feature_manifest_name}{name_suffix}", __file__)

    # Create config output path.
    config_path = get_output_path(
        "models", model_manifest_name, run_name, "configs", include_timestamp=False
    )

    for dataset in datasets:
        # Get data loading dataframe location.
        dataframe_location = get_dataframe_location_for_dataset(dataframe_manifest, dataset)
        dataframe_path = resolve_dataframe_location(dataframe_location)

        # Create evaluation config path.
        config_file = config_path / f"eval_{patch_type}_{dataset}{name_suffix}.yaml"

        # Build the evaluation config overrides.
        overrides = ModelConfigOverrideEval(
            model_manifest_name=model_manifest_name,
            eval_dataframe_path=dataframe_path,
            run_name=run_name,
            num_gpus=NUM_GPUS,
            num_workers=num_workers,
        )

        # Initialize the model with evaluation base config, apply overrides, and save config.
        cytodl_model = CytoDLModel()
        cytodl_model.load_config_from_dict(base_config)
        cytodl_model.override_config(overrides.to_dict(dataset, patch_type, name_suffix))
        cytodl_model.save_config(config_file)
        logger.info("Evaluation config saved to [ %s ]", config_file)

        # Populate manifest with evaluation run location and parameters.
        feature_manifest.parameters = {"patch_type": patch_type}
        feature_manifest.locations[dataset] = DataframeLocation()
        save_dataframe_manifest(feature_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
