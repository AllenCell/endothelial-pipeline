from endo_pipeline.cli import Datasets
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["eval_diffae_model", "diffae_features"]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    datasets: Datasets | None = None,
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    config_name: str | None = None,
    finetuned: bool = False,
) -> None:
    """
    Evaluate a trained DiffAE model on grid-based crops of images from given dataset(s).

    **Workflow output**

    For each specified dataset, this workflow produces a table of latent features obtained from a
    non-overlapping grid of crops of the processed images from each dataset. If ``upload_to_fms``
    is True, the prediction dataframe is saved as a parquet file locally and uploaded to FMS.
    The FMS ID of the uploaded file is then stored in the dataframe manifest corresponding to the
    specified model manifest and run name: ``{model_manifest_name}_{run_name}_grid``.

    **Config overrides**

    If ``config_name`` is provided, the model config loaded from the model manifest
    will be overridden with the specified config in ``src/configs/models``. If it is not provided,
    then the default DiffAE eval template config is used to override the loaded model config.
    The reason for doing this default override is that the training config by default does not
    contain settings for the ``predict_dataloaders`` used during inference.

    **Finetuned model**
    If ``finetuned`` is set to True, the default eval config used to override the loaded model
    config will be the finetuned model eval config (instead of the base model eval config).

    **Workflow demo**

    If demo mode is enabled, the model will only be evaluated on the first few
    timepoints of the first position of the first dataset.

    **Example usage**

    .. code-block:: bash

        endopipe -v -g 1 eval-diffae-grid --datasets 20250409_20X

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest to load the model from.
    run_name
        Name of the model run to evaluate. If None, uses the most recent run.
    datasets
        Optional, list of datasets or dataset collections to evaluate the model on.
    resolution_level
        Resolution level to at which to load images (zarr file format) at.
    upload_to_fms
        True to upload the prediction file for each dataset to FMS, False to only save locally.
    config_name
        Optional, name of the model config to use to override the loaded model config.
    finetuned
        If true, defaults to loading the finetuned model eval config.
    """
    import logging
    from pathlib import Path

    from endo_pipeline import DEMO_MODE, NUM_GPUS
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        load_dataset_config,
        load_model_config,
    )
    from endo_pipeline.library.model import (
        evaluate_model_on_grid_of_crops_from_one_dataset,
        load_model_for_inference,
        upload_prediction_dataframe_to_fms,
    )
    from endo_pipeline.library.model.image_loading import get_include_positions
    from endo_pipeline.manifests import get_feature_dataframe_manifest_name, load_model_manifest
    from endo_pipeline.settings import (
        DIFFAE_MODEL_EVAL_CONFIG,
        DIFFAE_MODEL_EVAL_FINETUNE_CONFIG,
        Z_SLICE_OFFSETS,
    )

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in datasets]

    # load model manifest
    model_manifest = load_model_manifest(model_manifest_name)

    # use specified config to override loaded model config if provided,
    # otherwise use default eval config (base or finetuned, depending on "finetuned" arg)
    if config_name is not None:
        name_of_config = config_name
    elif finetuned:
        name_of_config = DIFFAE_MODEL_EVAL_FINETUNE_CONFIG
    else:
        name_of_config = DIFFAE_MODEL_EVAL_CONFIG
    # load eval config as an OmegaConf object
    eval_config = load_model_config(name_of_config)

    # load model run_name from model manifest, override with eval config,
    # and make sure that "model_manifest_name" and "run_name" are stored in the config
    # as "experiment_name" and "run_name" for logging purposes
    model = load_model_for_inference(model_manifest, run_name, eval_config)

    # evaluate model on images from each dataset
    for dataset_config in dataset_config_list:
        # Get positions to include.
        only_include_positions = get_include_positions(dataset_config)

        # When running workflow in demo mode, only use the first position from each
        # dataset and first two timepoints to speed up the dataloading process (if
        # dataset is not timelapse, then only one timepoint is used). Otherwise, use
        # default frame start and stop values (i.e. all timepoints) and keep all
        # rows in the dataset CSV.
        if DEMO_MODE:
            frame_start = 0
            frame_stop = 1 if dataset_config.is_timelapse else 0
            only_include_positions = only_include_positions[0:1]
            logger.warning(
                "Workflow demo is enabled, only processing first few "
                "timepoints of the first position of dataset: [ %s ]",
                dataset_config.name,
            )
        else:
            frame_start = None
            frame_stop = None

        prediction_path = evaluate_model_on_grid_of_crops_from_one_dataset(
            model=model,
            dataset_config=dataset_config,
            resolution_level=resolution_level,
            z_slice_offsets=Z_SLICE_OFFSETS,
            frame_start=frame_start,
            frame_stop=frame_stop,
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
                    model_manifest, model.cfg.run_name, crop_pattern="grid"
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
