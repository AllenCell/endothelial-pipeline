# %% [markdown]
# # Create CytoDL model config

# %% [markdown]
"""
Create and save a new model config file from a `CytoDLModelConfig` object.

The config is saved to the `configs/models` directory with file name matching
the name of the dataset. If a config with the same name already exists, it will
be overwritten.

### Optional fields

Some fields in the config are optional, and will be set to a default value if
not provided. All optional fields are provided as commented lines of code. If an
optional field should be set, uncomment the corresponding line to set the value.
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from endo_pipeline.configs import CytoDLModelConfig, save_model_config

# %%
model = CytoDLModelConfig(
    # ============================ REQUIRED FIELDS =============================
    name="unique_model_name",
    mlflow_run_id="MLflow_run_id",
    # ============================ OPTIONAL FIELDS =============================
    # training_datasets=[
    #     "dataset_name1",
    #     "dataset_name2",
    # ]
)

save_model_config(model)

# %%
