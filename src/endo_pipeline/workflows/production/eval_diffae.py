from endo_pipeline.cli import CropPattern
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    crop_pattern: CropPattern,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> None:
    """
    Evaluate a DiffAE model using the provided configuration.

    #diffae #model-evaluation

    This workflow runs model evaluations based on configs produced by the
    `build-diffae-eval-config` workflow to calculate latent features. The
    workflow will only run the evaluation for datasets that do not already have
    features calculated (based on the corresponding dataframe manifest).

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe eval-diffae CROP_PATTERN -vd
    ```

    To run the workflow for a specific model manifest name and run name:

    ```bash
    uv run endopipe eval-diffae CROP_PATTERN \
        --model-manifest-name MODEL_MANIFEST_NAME \
        --run-name RUN_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will use the
    dataframe manifest and model config with the `_demo` suffix produced by
    also running `create-diffae-eval-dataframe` and `build-diffae-eval-config`
    in demo mode. The workflow will only evaluate the first dataset.

    Parameters
    ----------
    crop_pattern
        Crop pattern used for model evaluation.
    model_manifest_name
        Name for the model manifest to use for evaluation.
    run_name
        Name for the model run to use for evaluation.
    """

    import logging
    from pathlib import Path

    from cyto_dl.api import CytoDLModel

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.model.eval_model import (
        update_prediction_from_crops_with_metadata,
        update_prediction_from_tracks_with_metadata,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )

    logger = logging.getLogger(__name__)

    # Get available evaluation runs from given model manifest.
    model_manifest = load_model_manifest(model_manifest_name)
    name_suffix = "_demo" if DEMO_MODE else ""
    feature_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern
    )
    feature_manifest = load_dataframe_manifest(f"{feature_manifest_name}{name_suffix}")
    datasets = [key for key, loc in feature_manifest.locations.items() if loc.fmsid is None]

    if not datasets:
        logger.error("No pending evaluation runs in model manifest [ %s ]", model_manifest_name)
        raise ValueError(
            "No pending evaluation run configs available. "
            "Use build-diffae-eval-config to create new evaluation run configs."
        )

    # When running workflow in demo mode, only evaluate the first dataset.
    if DEMO_MODE:
        logger.warning("DEMO MODE - Only evaluating first dataset")
        datasets = datasets[:1]

    # Get config path based on model manifest and run name.
    name_suffix = "_demo" if DEMO_MODE else ""
    config_path = get_output_path(
        "models",
        model_manifest_name,
        run_name,
        "configs",
        include_timestamp=False,
        create_directories=False,
    )

    logger.info("Model manifest name: [ %s ]", model_manifest_name)
    logger.info("Run name: [ %s ]", run_name)

    for dataset in datasets:
        # Build evaluation config path.
        config_file = config_path / f"eval_{crop_pattern}_{dataset}{name_suffix}.yaml"

        # Initialize the model with evaluation config.
        cytodl_model = CytoDLModel()
        cytodl_model.load_config_from_file(config_file.as_posix())
        logger.info("Evaluation config loaded from [ %s ]", config_file)

        # Run prediction.
        cytodl_model.predict()

        # Get output information from model config.
        output_dir = Path(cytodl_model.cfg.paths.output_dir)
        filename_suffix = cytodl_model.cfg.callbacks.prediction_saver.save_suffix
        output_path = output_dir / f"predict_{filename_suffix}.parquet"

        # Add metadata to prediction output.
        if crop_pattern == "grid":
            crop_size = cytodl_model.cfg.model.spatial_inferer.splitter.patch_size
            update_prediction_from_crops_with_metadata(
                dataset_name=dataset,
                model_manifest_name=model_manifest_name,
                run_name=run_name,
                crop_size=crop_size,
                prediction_path=output_path,
            )
        elif crop_pattern == "tracked":
            update_prediction_from_tracks_with_metadata(
                dataset_name=dataset,
                model_manifest_name=model_manifest_name,
                run_name=run_name,
                prediction_path=output_path,
            )

        # Create location object with output path
        location = feature_manifest.locations.get(dataset, DataframeLocation())
        location.path = output_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            dataset_config = load_dataset_config(dataset)
            annotations = build_fms_annotations(
                dataset_config, model_manifest=model_manifest, run_name=run_name
            )
            fmsid = upload_file_to_fms(output_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None

        # Add dataframe location to dataframe manifest and save.
        feature_manifest.locations[dataset] = location
        save_dataframe_manifest(feature_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
