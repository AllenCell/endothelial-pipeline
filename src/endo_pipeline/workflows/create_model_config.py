# %% [markdown]
# # Create Model config

# %% [markdown]
"""
Create and save a new model config file from a `ModelConfig` object.

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
from src.endo_pipeline.configs import ModelConfig, ModelManifest, save_model_config

# %%
model = ModelConfig(
    # ============================ REQUIRED FIELDS =============================
    name="diffae_patch_128x128_2025-06-30",
    mlflow_run_id="09c2421d405d45e59ffc883fbd2f69a2",
    # ============================ OPTIONAL FIELDS =============================
    # manifest_fmsids=[
    #    ModelManifest(
    #        dataset_name="dataset_name",
    #        fmsid="FMS ID for the dataset manifest",
    #    )
    # ],
    # training_datasets=[
    #     "dataset_name1",
    #     "dataset_name2",
    # ]
    # train_manifest_fmsid="FMS ID for the training manifest dataset",
    # test_manifest_fmsid="FMS ID for the test manifest dataset",
)

save_model_config(model)

# %%
