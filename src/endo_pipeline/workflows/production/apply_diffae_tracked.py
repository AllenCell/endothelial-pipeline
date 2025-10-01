from endo_pipeline.cli import Datasets

TAGS = ["apply_diffae_model", "diffae_features"]


def main(
    model_manifest_name: str = "diffae_04_10",
    run_name: str | None = None,
    datasets: Datasets | None = None,
    upload_to_fms: bool = True,
    save_path: str | None = None,
    eval_config_path: str | None = None,
    user_overrides: str | dict | None = None,
) -> None:
    """
    Apply a trained DiffAE model to single-cell-track-based crops of images from multiple datasets.

    Produces a table of latent features from a crops centered on tracked cells for each dataset.

    **Workflow demo**

    If demo mode is enabled, the model will only be evaluated on the first position of the first
    of the specified datasets.

    **Eval config override**

    If ``eval_config_path`` is provided, the model config loaded from the model manifest
    will be overridden with the config from the specified path. If it is not provided,
    then the default DiffAE eval template config is used to override the loaded model config.
    The reason for doing this override is that the training config by default does not
    contain settings for the ``predict_dataloaders`` used during inference.

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest to load the model from.
    run_name
        Name of the model run to apply. If None, uses the most recent run.
    datasets
        List of datasets or dataset collections to load images from. If not
        provided, workflow runs on the ``20250319_20X`` dataset.
    upload_to_fms
        True to upload the prediction file for each dataset to FMS, False to only save locally.
    save_path
        Path to save the prediction file locally.
    eval_config_path
        Optional, path to the model eval config to use to override the loaded model config.
    user_overrides
        Optional user overrides to apply to the model config.

    Returns
    -------
    :
        Saves and/or updates a DataframeManifest with the prediction file for each dataset.
    """
    import logging
    from pathlib import Path

    from endo_pipeline import DEMO_MODE, NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_and_override_model_for_inference
    from endo_pipeline.library.model import (
        apply_model_on_tracked_crops_from_one_dataset,
        parse_eval_config_path,
        upload_prediction_dataframe_to_fms,
    )
    from endo_pipeline.library.model.image_loading import get_include_positions
    from endo_pipeline.manifests import get_model_location_for_run, load_model_manifest
    from endo_pipeline.settings import Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = ["20250319_20X"]

    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in datasets]

    # load model from manifest
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = list(model_manifest.locations.keys())[-1] if run_name is None else run_name
    model_location = get_model_location_for_run(model_manifest, run_name_)
    path_to_eval_config = parse_eval_config_path(eval_config_path)
    model = load_and_override_model_for_inference(model_location, path_to_eval_config)

    # apply model to each dataset
    for dataset_config in dataset_config_list:
        only_include_positions = get_include_positions(dataset_config)
        if DEMO_MODE:
            only_include_positions = only_include_positions[:1]
            logger.warning(
                "Workflow demo is enabled, only processing first few "
                "timepoints of the first position of dataset: [ %s ]",
                dataset_config.name,
            )

        prediction_path = apply_model_on_tracked_crops_from_one_dataset(
            model=model,
            run_name=run_name_,
            dataset_config=dataset_config,
            save_path=save_path,
            user_overrides=user_overrides,
            z_slice_offsets=Z_SLICE_OFFSETS,
            only_include_positions=only_include_positions,
            num_gpus=NUM_GPUS,
        )

        if upload_to_fms:
            # upload prediction file to FMS
            # Store FMS ID in dataframe manifest.
            upload_prediction_dataframe_to_fms(
                prediction_path,
                dataset_config,
                model_manifest,
                run_name_,
                dataframe_manifest_name=f"{model_manifest_name}_{run_name_}_tracked",
                workflow_name=Path(__file__).stem,
                workflow_parameters={"z_slice_offsets": Z_SLICE_OFFSETS},
            )

        if DEMO_MODE:
            # only apply model to the first dataset in demo mode
            break


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
