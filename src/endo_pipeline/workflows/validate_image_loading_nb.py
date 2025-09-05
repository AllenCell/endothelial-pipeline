# %%
from endo_pipeline.configs import (
    get_annotated_positions,
    get_annotated_timepoints_for_position,
    get_position_integer_from_zarr_file_path,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path
from endo_pipeline.library.model import (
    MultiDimImageDataset,
    build_zarr_image_loading_dataframe,
    get_exclude_frames,
    get_include_positions,
    get_z_slice_bounds_per_position,
)

# %%
dataset_config = load_dataset_config("20250319_20X")
assert dataset_config.center_z_plane is not None

resolution_level = 2
channel = [0, 1]
frame_start = 0
frame_stop = 450
z_stack_offsets = (5, 15)
slice_by_global_center = True

# %%
z_slice_bounds_per_position = get_z_slice_bounds_per_position(
    dataset_config, z_stack_offsets, slice_by_global_center
)
only_include_positions = get_include_positions(dataset_config)
exclude_frames_by_position = get_exclude_frames(dataset_config)

# %%
# get list of all positions with annotations for artifact detection
annotated_positions = get_annotated_positions(dataset_config)

# verify that z-slice bounds are correctly calculated
# and that expected positions are included/excluded
# and that excluded timepoints are correctly identified/stored
for position in dataset_config.zarr_positions:
    z_slice_bounds = z_slice_bounds_per_position[position]
    global_center = dataset_config.center_z_plane[position]
    assert z_slice_bounds["z_start"] == max(0, global_center - z_stack_offsets[0])
    assert z_slice_bounds["z_stop"] == min(24, global_center + z_stack_offsets[1])

    if position not in annotated_positions:
        assert position in only_include_positions
    else:
        assert position not in only_include_positions

    annotated_timepoints = get_annotated_timepoints_for_position(dataset_config, position)

    assert annotated_timepoints == exclude_frames_by_position.get(position, [])

    print("Validated position:", position)

# %%
# build dataframe from parsed loading inputs
df = build_zarr_image_loading_dataframe(
    dataset_config=dataset_config,
    resolution_level=resolution_level,
    channel=channel,
    frame_start=frame_start,
    frame_stop=frame_stop,
    z_slice_bounds_per_position=z_slice_bounds_per_position,
    only_include_positions=only_include_positions,
    exclude_frames=exclude_frames_by_position,
)

# print a few rows to get a sense of what this dataframe is
df.head()

# save locally to pass path to MultiDimImageDataset
output_directory = get_output_path("dataframes")
dataframe_file_path = output_directory / "test_dataframe.parquet"
df.to_parquet(dataframe_file_path)

# %%
# build image dataset from dataframe
image_dataset = MultiDimImageDataset(dataframe_path=dataframe_file_path)

# print loading args for first image in dataset
print(image_dataset.data[0])

# %%
# loop over list of dicts that store image loading args
# for each image in the dataset (position and timepoint)
# and verify that the dataframe has been parsed as expected
current_position = -1
for image_loading_args in image_dataset.data:
    assert image_loading_args["dimension_order_out"] == "CZYX"
    assert image_loading_args["C"] == channel
    assert image_loading_args["resolution"] == resolution_level
    position = get_position_integer_from_zarr_file_path(image_loading_args["original_path"])
    if position != current_position:
        current_position = position
        print(f"Validating position: {position}")
    exclude_frames = get_annotated_timepoints_for_position(dataset_config, position)
    assert image_loading_args["T"] not in exclude_frames
    z_slice_bounds = z_slice_bounds_per_position[position]
    z_slice_list = list(range(z_slice_bounds["z_start"], z_slice_bounds["z_stop"] + 1))
    assert image_loading_args["Z"] == z_slice_list

# %%
