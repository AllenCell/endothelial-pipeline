import logging

from .model_config import ModelConfig, ModelManifest

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
