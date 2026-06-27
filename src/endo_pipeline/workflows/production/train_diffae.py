def main(model_manifest_name: str, run_name: str | None = None) -> None:
    """
    Train a DiffAE model using the provided configuration.

    #diffae #model-training

    This workflow produces a trained DiffAE model that is logged to MLflow and
    tracked in a ModelManifest object.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe train-diffae MODEL_MANIFEST_NAME -d
    ```

    To run the workflow for given model manifest and run name:

    ```bash
    uv run endopipe train-diffae MODEL_MANIFEST_NAME RUN_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will use the
    dataframe manifest and model config with the `_demo` suffix produced by
    also running `create-diffae-train-dataframe` and `build-diffae-train-config`
    in demo mode.

    Parameters
    ----------
    model_manifest_name
        Name for the model manifest to use for training.
    run_name
        Name for the model run to use for training.
    """

    import logging

    from cyto_dl.api import CytoDLModel

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import get_output_path
    from endo_pipeline.manifests import ModelLocation, load_model_manifest, save_model_manifest

    logger = logging.getLogger(__name__)

    if DEMO_MODE:
        logger.warning("DEMO MODE - Training with demo config")

    # Get available training runs from given model manifest.
    name_suffix = "_demo" if DEMO_MODE else ""
    model_manifest_name = f"{model_manifest_name}{name_suffix}"
    model_manifest = load_model_manifest(model_manifest_name)
    available_runs = [key for key, loc in model_manifest.locations.items() if loc.mlflowid is None]

    if not available_runs:
        logger.error("No pending training runs in model manifest [ %s ]", model_manifest_name)
        raise ValueError(
            "No pending training run configs available. "
            "Use build-diffae-train-config to create a new training run config."
        )

    # Select first available pending training run, if run name is not given.
    # Otherwise, make sure the requested run is an available pending run.
    if run_name is None:
        run_name = available_runs[0]
    elif run_name not in available_runs:
        logger.error(
            "Requested training run [ %s ] not found in model manifest [ %s ]",
            model_manifest_name,
            run_name,
        )
        raise ValueError(
            "No matching training run configs available. "
            "Use build-diffae-train-config to create a new training run config."
        )

    logger.info("Model manifest name: [ %s ]", model_manifest_name)
    logger.info("Run name: [ %s ]", run_name)

    # Build training config path.
    config_path = get_output_path(
        "models",
        model_manifest_name,
        run_name,
        "configs",
        include_timestamp=False,
        create_directories=False,
    )
    config_file = config_path / f"train{name_suffix}.yaml"

    # Initialize the model with training config.
    cytodl_model = CytoDLModel()
    cytodl_model.load_config_from_file(config_file)
    logger.info("Training config loaded from [ %s ]", config_file)

    # Run training.
    _, object_dict = cytodl_model.train()

    # Retrieve MLflow run ID
    mlflow_logger = object_dict["logger"][0]
    mflow_run_id = mlflow_logger.run_id
    logger.info("MLflow run ID [ %s ]", mflow_run_id)

    # Save MLflow run ID to model manifest
    model_manifest.locations[run_name] = ModelLocation(mlflowid=mflow_run_id)
    save_model_manifest(model_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
