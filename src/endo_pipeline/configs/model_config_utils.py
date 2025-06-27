import logging

from .dataset_config import DatasetConfig
from .dataset_config_io import get_available_dataset_names, load_all_dataset_configs
from .model_config import ModelConfig, ModelManifest
from .model_config_io import load_single_model_config

logger = logging.getLogger(__name__)


def get_model_manifest(dataset_config: DatasetConfig, model_config: ModelConfig) -> ModelManifest:
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
    # the input dataset_config.name
    for manifest in model_config.manifest_fmsids:
        if manifest.dataset_name == dataset_config.name:
            return manifest

    # if no manifest found, raise an error
    logger.error(
        "No manifest found for dataset %s in model config %s",
        dataset_config.name,
        model_config.name,
    )
    raise FileNotFoundError(
        f"No manifest found for dataset {dataset_config.name} in model config {model_config.name}"
    )


def load_datasets_with_model_manifest(
    model_name: str = "diffae_04_10",
    verbose: bool = False,
    timelapse_only: bool = False,
) -> list:
    """
    List all dataset names that have manifest data
    for a given model.
    """
    all_datasets = get_available_dataset_names()

    if verbose:
        if timelapse_only:
            print(f"Available timelapse datasets with {model_name} manifest data: ")
        else:
            print(f"Available datasets with {model_name} manifest data: ")
    dataset_list = []
    all_datasets = load_all_dataset_configs()
    for dataset_info in all_datasets:
        # get time_interval_in_minutes - any dataset
        # that is fixed or is a 20X/40X pair has default
        # time_interval_in_minutes of -1.0, so we skip
        time_interval_in_minutes = dataset_info.time_interval_in_minutes
        if timelapse_only and time_interval_in_minutes < 0:
            continue
        else:
            # this will throw an error if the manifest is not found
            try:
                model_config = load_single_model_config(model_name)
                model_manifest = get_model_manifest(dataset_info, model_config)
            except:
                model_manifest = None
        if model_manifest is not None:
            dataset_list.append(dataset_info)
            if verbose:
                print(f" - {dataset_info.name}")
    return dataset_list
