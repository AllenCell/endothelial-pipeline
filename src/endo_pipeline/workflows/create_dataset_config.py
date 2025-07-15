# %% [markdown]
# # Create dataset config

# %% [markdown]
"""
Create and save a new dataset config file from a `DatasetConfig` object.

The config is saved to the `configs/datasets` directory with file name matching
the name of the dataset. If a config with the same name already exists, it will
be overwritten.

### Fields requiring specific values

Some fields in the config will only allow specific values, as defined in the
`DatasetConfig` by the `Literal` type. For convenience, all available options
for these fields are provided as lines of code that can be commented or
uncommented as needed. Setting an invalid value will produce an error.

### Optional fields

Some fields in the config are optional, and will be set to a default value if
not provided. All optional fields are provided as commented lines of code. If an
optional field should be set, uncomment the corresponding line to set the value.
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from src.endo_pipeline.configs import DatasetConfig, ValidTimepoints, save_dataset_config

# %%
dataset = DatasetConfig(
    # ============================ REQUIRED FIELDS =============================
    name="unique_dataset_name",
    original_path="/path/to/original/dataset",
    zarr_path="//allen/aics/endothelial/morphological_features/image_data/converted_zarrs/DATE_FMSID",
    fmsid="FMSID",
    barcode="labkey_barcode",
    cell_lines=["AICS-126"],
    live_or_fixed_sample="live",
    # live_or_fixed_sample="fixed",
    is_timelapse=True,
    # live_or_fixed_sample="fixed-methanol",
    microscope="3i",
    # microscope="Nikon",
    shear_stress_regime="shear_stress_regime",
    use_cases=[
        # "classic_segmentation",
        # "feasibility",
        # "immunofluorescence",
        # "model_training",
        # "measurement",
        # "nuclear_label_free_predictions",
    ],
    pixel_size_xy_in_um=0.382,  # 3i 20X
    duration=0,
    time_interval_in_minutes=0.0,
    flow=[(0, 0, 0.0)],
    n_total_positions=0,
    channel_488_index=0,
    brightfield_channel_index=1,
    # ============================ OPTIONAL FIELDS =============================
    # flow_conditions=[[0,0,0.0]],
    # channel_405_index=0,
    # channel_561_index=0,
    # channel_640_index=0,
    # nuclear_label_free_seg_path="/path/to/nuclear/label/free/segmentation",
    # nuclear_stain_seg_path="/path/to/nuclear/stain/segmentation",
    # nuclear_seg_manifest_fmsid="FMS ID for nuclear segmentation manifest",
    # tracking_integration_fmsid="FMS ID for tracking integration",
    # diffae_tracking_integration_fmsid="FMS ID for diffusion autoencoder tracking integration",
    # is_reference=False,
    # valid_timepoints=ValidTimepoints(start=[0], stop=[0]),
    # cell_mean_features="FMS ID for cell mean features",
    # include_scenes=[0, 0, 0],
    # notes="",
)

save_dataset_config(dataset)

# %%
