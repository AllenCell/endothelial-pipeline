# %%
from bioio.writers import ome_zarr_writer_2 as ome_zarr_writer

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_image
from endo_pipeline.library.process.convert_to_zarr.write_zarr import (
    DEFAULT_XY_SCALING,
    DEFAULT_Z_SCALING,
    get_level_shapes,
    get_zarr_chunk_dims,
)
from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

# %%
dataset_name = "20241120_20X"
seg_type = "nuclear_labelfree_seg"
position = 0

# %%
dataset_config = load_dataset_config(dataset_name)
image_manifest = load_image_manifest(seg_type)
image_0 = get_image_location_for_dataset(image_manifest, dataset_name, position, 0)
image_location_list = [
    get_image_location_for_dataset(image_manifest, dataset_name, position, t).path
    for t in range(dataset_config.duration)
]

# %%
date = dataset_name.split("_")[0]
full_zarr_path = get_output_path(
    "zarr_conversion",
    f"{date}_{dataset_config.fmsid}",
    f"{date}_{dataset_config.fmsid}_P{position}.ome.zarr",
)

# %%
img = load_image(image_0)
im_shape = img.shape
dtype = img.dtype
# %%
interval_min = dataset_config.time_interval_in_minutes
if interval_min is None:
    interval_min = -1

zarr_chunk_dims_tuples = get_zarr_chunk_dims(
    (dataset_config.duration, 1, 1, img.shape[0], img.shape[1]),
    DEFAULT_XY_SCALING,
    DEFAULT_Z_SCALING,
)
writer = ome_zarr_writer.OmeZarrWriter()
writer.init_store(
    output_path=str(full_zarr_path),
    shapes=get_level_shapes(
        (dataset_config.duration, 1, 1, img.shape[0], img.shape[1]),
        DEFAULT_XY_SCALING,
        DEFAULT_Z_SCALING,
    ),
    chunk_sizes=zarr_chunk_dims_tuples,
    dtype=dtype,
)
# %%
# Write image sequence in batches
writer.write_t_batches_image_sequence(
    paths=image_location_list,
    channels=None,
    tbatch=4,
)
# %%
xy_scale = 0.3820158766750814
physical_scale = {
    "c": 1.0,  # default value for channel
    "t": interval_min,
    "z": 1.0,
    "y": xy_scale,
    "x": xy_scale,
}
physical_units = {
    "x": "micrometer",
    "y": "micrometer",
    "z": "micrometer",
    "t": "minute",
}

print(f"Physical dimensions: {physical_scale}")

meta = writer.generate_metadata(
    image_name=f"{dataset_name}_{position}",
    channel_names=["NUC_SEG"],
    physical_dims=physical_scale,
    physical_units=physical_units,
    channel_colors=[0xFFFFFF],
)
print("Writing metadata...")
writer.write_metadata(meta)

# %%
