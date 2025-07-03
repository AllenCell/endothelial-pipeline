# %% [markdown]
# # Validate dataset configs

# %% [markdown]
"""
Validate all existing ML models by checking config schemas and loading checkpoint files.

For each dataset config in the `configs/models` directory, confirm:

- All dataset configs follow the schema defined by `ModelConfig`
- All MLflow run IDs exist and can be opened (load checkpoint)
- All datasets in the ModelManifest are valid (have a DatasetConfig)
    and can be loaded (load manifest from the model via FMS)
- All datasets in the training_datasets list have a DatasetConfig

"""
# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
import logging

from cellsmap.util.manifest_io import get_dataframe_by_fmsid
from src.endo_pipeline.configs import (
    get_available_model_names,
    load_dataset_config,
    load_model_config,
    validate_model_config,
)
from src.endo_pipeline.library.model.mlflow import get_ckpt_path

# %%
default_tracking_uri = "https://production.int.allencell.org/mlflow/"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %%
for name in get_available_model_names():
    logger.info(f"Running validation for model [ {name} ]")

    # Validate dataset config schema.
    validate_model_config(name)

    # Load dataset config.
    model_config = load_model_config(name)

    # Check if model exists in MLFlow.
    try:
        model_ckpt = get_ckpt_path(model_config.mlflow_run_id, default_tracking_uri)
    except Exception as e:
        logger.error(
            "Failed to find checkpoint for model [ %s ] from MLflow run ID [ %s ]: %s",
            name,
            model_config.mlflow_run_id,
            str(e),
        )
        raise

    # Check if all datasets with manifests for this model
    # Have a DatasetConfig and can be loaded from FMS
    manifest_fmsids = model_config.manifest_fmsids
    logger.info("Validating manifests...")
    for dataset_manifest in manifest_fmsids:
        dataset_name = dataset_manifest.dataset_name

        # Load dataset config
        try:
            logger.debug("Loading dataset config for [ %s ]", dataset_name)
            # This will raise an error if the dataset does not exist
            # or is not valid.
            dataset_config = load_dataset_config(dataset_name)
        except FileNotFoundError:
            logger.error(
                "Failed to load dataset config for [ %s ]",
                dataset_name,
            )
            raise

        # Check if manifests can be loaded by the given FMSID
        try:
            df = get_dataframe_by_fmsid(dataset_manifest.fmsid)
        except FileNotFoundError:
            logger.error(
                "Failed to load manifest for dataset [ %s ] with FMSID [ %s ]",
                dataset_name,
                dataset_manifest.fmsid,
            )
            raise

    # Check if all training datasets have a DatasetConfig
    logger.info("Validating training datasets...")
    for dataset_name in model_config.training_datasets:
        logger.debug("Validating dataset [ %s ]", dataset_name)

        # Load dataset config
        try:
            logger.debug("Loading dataset config for [ %s ]", dataset_name)
            # This will raise an error if the dataset does not exist
            # or is not valid.
            dataset_config = load_dataset_config(dataset_name)
        except FileNotFoundError:
            logger.error(
                "Failed to load dataset config for [ %s ]",
                dataset_name,
            )
            raise


# %%
