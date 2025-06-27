# %% [markdown]
# # Validate dataset configs

# %% [markdown]
"""
This workflow is used to validate all existing dataset configs by confirming the
following:

- All dataset configs follow the schema defined by `DatasetConfig`
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
import logging

from src.endo_pipeline.configs import validate_all_dataset_configs

# %%
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %%
validate_all_dataset_configs()
