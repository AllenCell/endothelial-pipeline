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
from src.endo_pipeline.configs import (
    ChannelIndices,
    DatasetConfig,
    ValidTimepoints,
    save_dataset_config,
)

# %%
dataset = DatasetConfig(
    # ============================ REQUIRED FIELDS =============================
    name="unique_dataset_name",
    original_path="/path/to/original/dataset",
    zarr_path="//allen/aics/endothelial/morphological_features/image_data/converted_zarrs/DATE_FMSID",
    zarr_positions=[],
    fmsid="FMSID",
    barcode="labkey_barcode",
    cell_lines=["AICS-126"],
    live_or_fixed_sample="live",
    # live_or_fixed_sample="fixed",
    # live_or_fixed_sample="fixed-methanol",
    is_timelapse=True,
    microscope="3i",
    # microscope="Nikon",
    objective="20X",
    # objective="40X",
    shear_stress_regime="shear_stress_regime",
    pixel_size_xy_in_um=0.382,  # 3i 20X
    duration=0,
    time_interval_in_minutes=0.0,
    flow=[(0, 0, 0.0)],
    n_total_positions=0,
    original_channels=ChannelIndices(
        brightfield=1,
        channel_488=0,
    ),
    zarr_channels=ChannelIndices(
        brightfield=1,
        channel_488=0,
    ),
    # ============================ OPTIONAL FIELDS =============================
    # flow_conditions=[[0,0,0.0]],
    # valid_timepoints=ValidTimepoints(start=[0], stop=[0]),
    # include_scenes=[0, 0, 0],
    # notes="",
    # exclude_timepoints={
    #     0: [],
    #     1: [],
    #     2: [],
    #     3: [],
    #     4: [],
    #     5: []
    # },
    # center_z_plane={
    #     0: 10,
    #     1: 10,
    #     2: 10,
    #     3: 10,
    #     4: 10,
    #     5: 10
    # }
)

save_dataset_config(dataset)

# %%
