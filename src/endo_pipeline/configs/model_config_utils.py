import logging

from .dataset_config_io import load_single_dataset_config
from .model_config import ModelConfig, ModelManifest
from .model_config_io import load_single_model_config

logger = logging.getLogger(__name__)


def add_model_manifest(dataset_name: str, fmsid: str, model_config: ModelConfig) -> ModelConfig:
    """
    Add a model manifest to the model configuration.

    Inputs:
    - dataset_name: str, name of the dataset
    - fmsid: str, FMS ID of the manifest file
    - model_config: ModelConfig, configuration of the model

    Outputs:
    - ModelConfig, updated with the new manifest
    """

    # build a ModelManifest object
    add_model_manifest = ModelManifest(
        dataset_name=dataset_name,
        fmsid=fmsid,
    )

    # add the manifest to the model config
    manifest_fmsids = model_config.manifest_fmsids
    if manifest_fmsids is None:
        manifest_fmsids = []
    manifest_fmsids.append(add_model_manifest)
    model_config.manifest_fmsids = manifest_fmsids

    # return the updated model config
    return model_config


def get_model_manifest(dataset_name: str, model_config: ModelConfig) -> ModelManifest:
    """
    Get model manifest for a given dataset and model configuration.

    Inputs:
    - dataset_name: str, name of the dataset
    - model_config: ModelConfig, configuration of the model

    Outputs:
    - ModelManifest, containing dataset name and fmsid
    """
    if model_config.manifest_fmsids is None:
        logger.error("No manifests for model config %s", model_config.name)
        raise FileNotFoundError(f"No manifest fmsids found in model config {model_config.name}")

    # search the ModelConfig.manifest_fmsids for the
    # ModelManifest element with dataset_name matching
    # the input dataset_name
    for manifest in model_config.manifest_fmsids:
        if manifest.dataset_name == dataset_name:
            return manifest

    # if no manifest found, raise an error
    logger.error(
        "No manifest found for dataset %s in model config %s", dataset_name, model_config.name
    )
    raise FileNotFoundError(
        f"No manifest found for dataset {dataset_name} in model config {model_config.name}"
    )


def list_datasets_with_model_manifest(
    model_name: str = "diffae_04_10",
    verbose: bool = False,
    timelapse_only: bool = False,
) -> list:
    """
    List all dataset names that have manifest data
    for a given model.
    """
    model_manifest = load_single_model_config(model_name).manifest_fmsids
    dataset_list = []

    # if model_manifest is None, return an empty list
    if model_manifest is None:
        logger.warning("No manifest fmsids found for model [ %s ]", model_name)
        return dataset_list

    # else, loop through the model_manifest
    for manifest in model_manifest:
        dataset_name = manifest.dataset_name
        # get time_interval_in_minutes - any dataset
        # that is fixed or is a 20X/40X pair has default
        # time_interval_in_minutes of -1.0, so we skip
        time_interval_in_minutes = load_single_dataset_config(dataset_name).time_interval_in_minutes
        if timelapse_only and time_interval_in_minutes < 0:
            continue

        # add the dataset name to the list
        # if verbose, print the dataset name
        dataset_list.append(dataset_name)

    # if verbose, log the dataset names
    if verbose:
        if timelapse_only:
            logger.info("Timelapse datasets with manifest data for model [ %s ]:", model_name)
        else:
            logger.info("All datasets with manifest data for model [ %s ]:", model_name)
        logger.info("\n".join(dataset_list))

    return dataset_list
