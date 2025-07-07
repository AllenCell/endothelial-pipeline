# %% [markdown]
# # Validate model configs

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

from src.endo_pipeline.configs import (
    get_available_model_names,
    load_dataset_config,
    load_model_config,
    validate_model_config,
)
from src.endo_pipeline.io import load_dataframe_from_fms
from src.endo_pipeline.library.model.mlflow import get_ckpt_path

# %%
DEFAULT_TRACKING_URI = "https://production.int.allencell.org/mlflow/"

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
        model_ckpt = get_ckpt_path(model_config.mlflow_run_id, DEFAULT_TRACKING_URI)
    except FileNotFoundError:
        logger.error(
            "Failed to find checkpoint for model [ %s ] from MLflow run ID [ %s ]",
            name,
            model_config.mlflow_run_id,
        )
        raise
    except ValueError:
        logger.error(
            "Failed to load model [ %s ] from MLflow run ID [ %s ]",
            name,
            model_config.mlflow_run_id,
        )
        raise

    # Check if all datasets with manifests for this model
    # Have a DatasetConfig and can be loaded from FMS
    manifest_fmsids = model_config.manifest_fmsids
    logger.info("Validating manifests...")
    for dataset_manifest in manifest_fmsids:
        dataset_name = dataset_manifest.dataset_name

        # Load dataset config
        dataset_config = load_dataset_config(dataset_name)

        # load dataframe from FMS
        df = load_dataframe_from_fms(dataset_manifest.fmsid)

    # Check if all training datasets have a DatasetConfig
    logger.info("Validating training datasets...")
    for dataset_name in model_config.training_datasets:
        logger.debug("Validating dataset [ %s ]", dataset_name)

        # Load dataset config
        dataset_config = load_dataset_config(dataset_name)


# %%
