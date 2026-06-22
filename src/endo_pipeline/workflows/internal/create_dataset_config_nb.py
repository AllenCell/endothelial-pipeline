"""
Create and save a new dataset config file from a `DatasetConfig` object.

#internal #datasets

This notebooks save the dataset config to the `configs/datasets` directory with
file name matching the name of the dataset. If a config with the same name
already exists, it will be overwritten.

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
from endo_pipeline.configs import PositionAnnotation  # noqa: F401, I001
from endo_pipeline.configs import TimepointAnnotation  # noqa: F401
from endo_pipeline.configs import (
    ChannelIndices,
    ChannelName,
    DatasetConfig,
    FlowCondition,
    ShearStressRegime,
    save_dataset_config,
    validate_dataset_config,
)
from endo_pipeline.manifests import (
    add_image_location_to_manifest,
    load_image_manifest,
    save_image_manifest,
)

# %% Create new dataset config

dataset_config = DatasetConfig(
    # ============================ REQUIRED FIELDS =============================
    name="unique_dataset_name",
    date="YYYYMMDD",
    original_path="/path/to/original/dataset",
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
    shear_stress_regime=[ShearStressRegime.NO],
    # shear_stress_regime=[ShearStressRegime.MIN],
    # shear_stress_regime=[ShearStressRegime.LOW],
    # shear_stress_regime=[ShearStressRegime.MEDIUM],
    # shear_stress_regime=[ShearStressRegime.HIGH],
    # shear_stress_regime=[ShearStressRegime.MAX],
    pixel_size_xy_in_um=0.382,  # 3i 20X
    duration=0,
    time_interval_in_minutes=5.0,
    n_total_positions=6,
    original_channel_indices=ChannelIndices(
        brightfield=1,
        channel_488=0,
    ),
    zarr_channel_indices=ChannelIndices(
        brightfield=1,
        channel_488=0,
    ),
    channel_names=["EGPF", ChannelName.BF],
    flow_conditions=[
        FlowCondition(start=0, stop=576, shear_stress=0.0),
    ],
    # ============================ OPTIONAL FIELDS =============================
    # include_scenes=[0, 0, 0],
    # notes="",
    # position_annotations = {
    #     PositionAnnotation.DUST_ARTIFACT: [2, 3, 4]
    # },
    #     timepoint_annotations={
    #         TimepointAnnotation.BF_SCOPE_ERROR: {
    #             0: [],
    #             1: [],
    #             2: [],
    #             3: [],
    #             4: [],
    #             5: []
    #         },
    #         TimepointAnnotation.BF_TEMP_ARTIFACT: {
    #             0: [],
    #             1: [],
    #             2: [],
    #             3: [],
    #             4: [],
    #             5: []
    #         },
    #         TimepointAnnotation.CELL_PILING: {
    #             0: [(356, 550)],
    #             1: [(356, 550)],
    #             2: [(356, 550)],
    #             3: [(356, 550)],
    #             4: [(356, 550)],
    #             5: [(356, 550)]
    #         },
    #         TimepointAnnotation.NOT_STEADY_STATE: {
    #             0: [(0, 100)],
    #             1: [(0, 100)],
    #             2: [(0, 100)],
    #             3: [(0, 100)],
    #             4: [(0, 100)],
    #             5: [(0, 100)]
    #         },
    #         TimepointAnnotation.XY_SHIFT: {
    #             0: [],
    #             1: [],
    #             2: [],
    #             3: [],
    #             4: [],
    #             5: []
    #         },
    #         TimepointAnnotation.Z_SHIFT: {
    #             0: [],
    #             1: [],
    #             2: [],
    #             3: [],
    #             4: [],
    #             5: []
    #         },
    #         TimepointAnnotation.UNFED: {
    #             0: [(118, 256)],
    #             1: [(118, 256)],
    #             2: [(118, 256)],
    #             3: [(118, 256)],
    #             4: [(118, 256)],
    #             5: [(118, 256)]
    #         },
    #     }
)

save_dataset_config(dataset_config)
validate_dataset_config(dataset_config.name)

# %% Update image manifest with new dataset location

zarr_parent_path = "//allen/aics/endothelial/morphological_features/image_data/converted_zarrs/"
image_manifest = load_image_manifest("image_zarr")
add_image_location_to_manifest(image_manifest, dataset_config, zarr_parent_path)
save_image_manifest(image_manifest)

# %%
print("Reminder to add dataset to relevant collections!")
# %%
