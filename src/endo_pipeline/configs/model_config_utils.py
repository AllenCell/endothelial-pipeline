import logging

from .dataset_config_io import get_datasets_in_collection, load_dataset_config
from .model_config import ModelConfig, ModelManifest
from .model_config_io import load_model_config, save_model_config

logger = logging.getLogger(__name__)


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
    for model_manifest in model_config.manifest_fmsids:
        if model_manifest.dataset_name == dataset_name:
            return model_manifest

    # if no manifest found, raise an error
    logger.error(
        "No manifest found for dataset %s in model config %s", dataset_name, model_config.name
    )
    raise FileNotFoundError(
        f"No manifest found for dataset {dataset_name} in model config {model_config.name}"
    )


def add_model_manifest(model_config: ModelConfig, dataset_name: str, fmsid: str) -> ModelConfig:
    """
    Add a model manifest to the model configuration.

    Inputs:
    - model_config: ModelConfig, configuration of the model
    - dataset_name: str, name of the dataset
    - fmsid: str, fmsid of the model manifest for the dataset

    Outputs:
    - ModelConfig, updated model configuration with the new manifest added
    """

    if model_config.manifest_fmsids is None:
        model_config.manifest_fmsids = []

    # check if a manifest already exists for this dataset
    if any(manifest.dataset_name == dataset_name for manifest in model_config.manifest_fmsids):
        logger.warning(
            "Manifest for dataset %s already exists in model config %s, overwriting it.",
            dataset_name,
            model_config.name,
        )

    # create a new ModelManifest and add it to the model_config
    new_manifest = ModelManifest(dataset_name=dataset_name, fmsid=fmsid)
    model_config.manifest_fmsids.append(new_manifest)

    return model_config


def get_timelapse_model_manifests(model_config: ModelConfig) -> list[ModelManifest]:
    """
    Get the list of model manifests that are timelapse datasets.

    Inputs:
    - model_config: ModelConfig, configuration of the model

    Outputs:
    - list of ModelManifest, containing only timelapse datasets
    """
    if len(model_config.manifest_fmsids) == 0:
        logger.error("No manifests for model config %s", model_config.name)
        raise FileNotFoundError(f"No manifest fmsids found in model config {model_config.name}")

    # filter manifests to only include timelapse datasets
    timelapse_manifest_list = []
    for model_manifest in model_config.manifest_fmsids:
        data_config = load_dataset_config(model_manifest.dataset_name)
        if data_config.time_interval_in_minutes is None:
            continue
        timelapse_manifest_list.append(model_manifest)

    return timelapse_manifest_list


def get_pca_reference_model_manifests(model_config: ModelConfig) -> list[ModelManifest]:
    """
    Get the list of model manifests that are reference datasets for PCA.

    Inputs:
    - model_config: ModelConfig, configuration of the model

    Outputs:
    - list of ModelManifest, containing only reference datasets for PCA
    """

    # load data configs to get reference datasets
    reference_dataset_name_list = get_datasets_in_collection("reference")

    # list of model manifests
    model_manifests = []
    for dataset_name in reference_dataset_name_list:
        # check if the dataset is in the model config
        try:
            model_manifests.append(get_model_manifest(dataset_name, model_config))
        except FileNotFoundError:
            logger.warning(
                "Do not have manifests for all PCA reference datasets in model config %s.",
                model_config.name,
            )
            continue
    if len(model_manifests) == 0:
        logger.error("No reference datasets found for PCA in model config %s.", model_config.name)
        raise FileNotFoundError("Insufficient reference datasets for PCA.")
    return model_manifests
