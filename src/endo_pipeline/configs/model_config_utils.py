import logging

from src.endo_pipeline.configs import (
    CytoDLModelConfig,
    ModelManifest,
    get_datasets_in_collection,
    load_dataset_config,
)

logger = logging.getLogger(__name__)


def add_model_manifest(
    model_config: CytoDLModelConfig, dataset_name: str, fmsid: str
) -> CytoDLModelConfig:
    """
    Add a model manifest to the model configuration.

    Inputs:
    - model_config: CytoDLModelConfig, configuration of the model
    - dataset_name: str, name of the dataset
    - fmsid: str, fmsid of the model manifest for the dataset

    Outputs:
    - CytoDLModelConfig, updated model configuration with the new manifest added
    """

    if model_config.manifest_fmsids is None:
        model_config.manifest_fmsids = []

    # check if a manifest already exists for this dataset
    if any(manifest.dataset_name == dataset_name for manifest in model_config.manifest_fmsids):
        logger.warning(
            "Manifest for dataset [ %s ] already exists in model config [ %s ], "
            + "adding potential duplicate.",
            dataset_name,
            model_config.name,
        )

    # create a new ModelManifest and add it to the model_config
    new_manifest = ModelManifest(dataset_name=dataset_name, fmsid=fmsid)
    model_config.manifest_fmsids.append(new_manifest)

    return model_config


def get_labelfree_nuclei_prediction_model_name() -> str:
    """Get the name of the label-free nuclei prediction model."""

    return "nuc_pred_labelfree_finetuned_20250419"
