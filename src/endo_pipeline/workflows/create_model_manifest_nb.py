# %% [markdown]
# # Create Model Manifest

# %% [markdown]
"""
Create and save a new model manifest file from a ModelManifest object.

The config is saved to the `manifests/models` directory.
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from endo_pipeline.manifests import ModelLocation  # noqa: F401
from endo_pipeline.manifests import ModelManifest, save_model_manifest  # noqa: I001

# %%
model_manifest = ModelManifest(
    # ============================ REQUIRED FIELDS =============================
    name="demo_model_manifest",
    workflow="train-diffae",
    # ============================ OPTIONAL FIELDS =============================
    # parameters={"z_slice_offsets": None, "exclude_cell_piling": False},
    # locations={"20250920_run": ModelLocation(mlflowid="abcdefg")},
)

save_model_manifest(model_manifest)

# %%
