from endo_pipeline.cli import Datasets

TAGS = ["eval_diffae_model", "diffae_features"]


def main(
    model_manifest_name: str = "diffae_04_10",
    run_name: str | None = None,
    datasets: Datasets | None = None,
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    config_name: str | None = None,
    fintuned: bool = False,
) -> None:
    """
    Evaluate a trained DiffAE model on grid-based crops of images from given dataset(s).

    Produces a table of latent features from a non-overlapping grid of crops for each dataset.
    The model is applied at the specified resolution level.

    **Workflow demo**

    If demo mode is enabled, the model will only be evaluated on the first few
    timepoints of the first position of the first dataset.

    **Config overrides**

    If ``config_name`` is provided, the model config loaded from the model manifest
    will be overridden with the specified config in ``src/configs/models``. If it is not provided,
    then the default DiffAE eval template config is used to override the loaded model config.
    The reason for doing this default override is that the training config by default does not
    contain settings for the ``predict_dataloaders`` used during inference.

    **Finetuned model**
    If ``finetuned`` is set to True, the default eval config used to override the loaded model
    config will be the finetuned model eval config (instead of the base model eval config).

    **Example usage**

    .. code-block:: bash

        endopipe -vg eval-diffae-grid --datasets 20250409_20X

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest to load the model from.
    run_name
        Name of the model run to evaluate. If None, uses the most recent run.
    datasets
        List of datasets or dataset collections to load images from. If not
        provided, workflow runs on the ``live_20X_objective_3i_microscope``
        dataset collection.
    resolution_level
        Resolution level to at which to load images (zarr file format) at.
    upload_to_fms
        True to upload the prediction file for each dataset to FMS, False to only save locally.
    config_name
        Optional, name of the model config to use to override the loaded model config.
    finetuned
        If true, defaults to loading the finetuned model eval config.

    Returns
    -------
    :
        Saves and/or updates a DataframeManifest with the prediction file for each dataset.
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
    from endo_pipeline.manifests import load_model_manifest
    from endo_pipeline.settings import EVAL_CONFIG, Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in datasets]

    # load model manifest
    model_manifest = load_model_manifest(model_manifest_name)

    # use input path to an eval config if provided, else use path to diffae_eval.yaml
    name_of_config = config_name if config_name else EVAL_CONFIG
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
                dataframe_manifest_name=f"{model_manifest_name}_{model.cfg.run_name}_grid",
                workflow_name=Path(__file__).stem,
                workflow_parameters={"z_slice_offsets": Z_SLICE_OFFSETS},
            )

        if DEMO_MODE:
            # only eval on the first dataset in demo mode
            break


if __name__ == "__main__":

    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
