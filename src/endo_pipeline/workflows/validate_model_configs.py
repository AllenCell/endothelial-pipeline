# %% [markdown]
# # Validate model configs

# %% [markdown]
"""
Validate all existing ML models by checking config schemas and loading checkpoint files.
\
\
This script requires the extra `ml_workflows` dependencies to run.
If these dependencies are not installed, you can run the following command:
```
uv sync --extras ml_workflows
```
and then run this notebook. \
\
For each dataset config in the `configs/models` directory, confirm:

- All dataset configs follow the schema defined by `ModelConfig`.
- All MLflow run IDs exist and can be opened (load checkpoint).
- All datasets in the `training_datasets` list have a `DatasetConfig`.
"""  # noqa: D415, D400
# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")


# %%
import logging

from src.endo_pipeline.configs import (
    CytoDLModelConfig,
    get_available_model_names,
    load_dataset_config,
    load_model_config,
    validate_model_config,
)
from src.endo_pipeline.io import load_dataframe_from_fms

# try to import from a module that requires
# the ml_workflows extra dependencies
# if this fails, raise an ImportError with a helpful message
try:
    from src.endo_pipeline.library.model import get_ckpt_path
except ImportError as e:
    raise ImportError(
        "This notebook requires the `ml_workflows` extra dependencies to run. "
        "Please install them with `uv sync --extras ml_workflows` and try again."
    ) from e

# %%
DEFAULT_TRACKING_URI = "https://production.int.allencell.org/mlflow/"

# filter out INFO-level messages from srrc.endo_pipeline.io.input
logging.getLogger("src.endo_pipeline.io.input").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %%
for name in get_available_model_names():
    logger.info(f"Running validation for model [ {name} ]")

    # Validate dataset config schema.
    validate_model_config(name)

    # Load dataset config.
    model_config = load_model_config(name)

    # Skip remaining validation if model not a CytoDL model
    if not isinstance(model_config, CytoDLModelConfig):
        logger.info("Skipping remaining validation for non-CytoDL model [ %s ]", name)
        continue

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

    # Check if all training datasets have a DatasetConfig
    # catch raised error and log a warning instead:
    # right now, the `diffae_04_10` model has training
    # datasets that do not have a DatasetConfig
    # this workflow can be updated if we deprecate this model
    logger.info("Validating training datasets...")
    for dataset_name in model_config.training_datasets:
        logger.debug("Validating dataset [ %s ]", dataset_name)

        if name == "diffae_04_10" or name == "diffae_patch_64x64_2025-06-30":
            try:
                # Load dataset config
                dataset_config = load_dataset_config(dataset_name)
            except FileNotFoundError:
                logger.warning(
                    "Training dataset [ %s ] for model [ %s ] does not have a DatasetConfig",
                    dataset_name,
                    name,
                )
                continue
        else:
            # Load dataset config
            dataset_config = load_dataset_config(dataset_name)

    logger.info("Validation for model [ %s ] completed successfully", name)
# %%
