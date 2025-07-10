# %% [markdown]
# # Validate dynamics analysis pipeline configs

# %% [markdown]
"""
Validate all existing configs for running the 2D DiffAE feature dynamics analysis pipeline
by checking config schemas and loading checkpoint files.
\
\
For each dynamics config in the `configs/dynamics_pipeline` directory, confirm:
- All configs follow the schema defined by `DynamicsConfig`.
"""  # noqa: D415, D400
# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")


# %%
import logging

from src.endo_pipeline.configs import get_available_dynamics_configs, validate_dynamics_config

# %%
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %%
for config_name in get_available_dynamics_configs():
    logger.info(f"Running validation for [ {config_name} ] configuration")

    # Validate dynamics config schema.
    validate_dynamics_config(config_name)

    logger.info("Validation for model [ %s ] completed successfully", config_name)
# %%
