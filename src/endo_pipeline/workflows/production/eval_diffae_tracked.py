from endo_pipeline.cli import Datasets
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["eval_diffae_model", "diffae_features"]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    datasets: Datasets | None = None,
    upload_to_fms: bool = True,
    save_path: str | None = None,
    config_name: str | None = None,
) -> None:
    """
    Evaluate a trained DiffAE model on tracked-cell-based crops of images from given dataset(s).

    Produces a table of latent features from a crops centered on tracked cells for each dataset.

    **Workflow output**

    For each specified dataset, this workflow produces a table of latent features obtained from
    crops of the processed images from each dataset centered on tracked cells. If ``upload_to_fms``
    is True, the prediction dataframe is saved as a parquet file locally and uploaded to FMS.
    The FMS ID of the uploaded file is then stored in the dataframe manifest corresponding to the
    specified model manifest and run name: ``{model_manifest_name}_{run_name}_tracked``.

    **Config overrides**

    If ``config_path`` is provided, the model config loaded from the model manifest
    will be overridden with the specified config in ``src/configs/models``. If it is not provided,
    then the default DiffAE eval template config is used to override the loaded model config.
    The reason for doing this override is that the training config by default does not
    contain settings for the ``predict_dataloaders`` used during inference.

    **Workflow demo**

    If demo mode is enabled, the model will only be evaluated on the first position of the first
    of the specified datasets.

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest to load the model from.
    run_name
        Name of the model run to evaluate. If None, uses the most recent run.
    datasets
        List of datasets or dataset collections to load images from.
    upload_to_fms
        True to upload the prediction file for each dataset to FMS, False to only save locally.
    save_path
        Path to save the prediction file locally.
    config_name
        Optional, name of the model eval config to use to override the loaded model config.
    """
    import logging
    from pathlib import Path

    from endo_pipeline import DEMO_MODE, NUM_GPUS
    from endo_pipeline.configs import load_dataset_config, load_model_config
    from endo_pipeline.library.model import (
        evaluate_model_on_tracked_crops_from_one_dataset,
        load_model_for_inference,
        upload_prediction_dataframe_to_fms,
    )
    from endo_pipeline.library.model.image_loading import get_include_positions
    from endo_pipeline.manifests import get_feature_dataframe_manifest_name, load_model_manifest
    from endo_pipeline.settings import DIFFAE_MODEL_EVAL_CONFIG, Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = ["20250319_20X"]

    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in datasets]

    # load model manifest
    model_manifest = load_model_manifest(model_manifest_name)

    # use input path to an eval config if provided, else use path to diffae_eval.yaml
    name_of_config = config_name if config_name else DIFFAE_MODEL_EVAL_CONFIG
    # load eval config as an OmegaConf object
    eval_config = load_model_config(name_of_config)

    # load model run_name from model manifest, override with eval config,
    # and make sure that "model_manifest_name" and "run_name" are stored in the config
    # as "experiment_name" and "run_name" for logging purposes
    model = load_model_for_inference(model_manifest, run_name, eval_config)

    # evaluate model on images from each dataset
    for dataset_config in dataset_config_list:
        only_include_positions = get_include_positions(dataset_config)
        if DEMO_MODE:
            only_include_positions = only_include_positions[:1]
            logger.warning(
                "Workflow demo is enabled, only processing tracks from "
                "the first position of dataset: [ %s ]",
                dataset_config.name,
            )

        prediction_path = evaluate_model_on_tracked_crops_from_one_dataset(
            model=model,
            dataset_config=dataset_config,
            save_path=save_path,
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
                model.cfg.run_name,
                dataframe_manifest_name=get_feature_dataframe_manifest_name(
                    model_manifest, model.cfg.run_name, crop_pattern="tracked"
                ),
                workflow_name=Path(__file__).stem,
                workflow_parameters={"z_slice_offsets": Z_SLICE_OFFSETS},
            )

        if DEMO_MODE:
            # only eval on the first dataset in demo mode
            break


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
