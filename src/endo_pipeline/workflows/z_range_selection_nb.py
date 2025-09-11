# %%
from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.process.z_stack_selection import (
    plot_histogram_upper_slices_available,
    plot_normalized_profiles,
    visualize_z_slices_with_offsets,
)
from endo_pipeline.settings import LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET

# %%
datasets = get_datasets_in_collection("timelapse")

# %%
save_dir = get_output_path(
    "z_range_selection", f"images_offsets_{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}"
)
for dataset in datasets:
    dataset_config = load_dataset_config(dataset)
    for position in dataset_config.zarr_positions:
        visualize_z_slices_with_offsets(dataset_config, position, 0, save_dir)

# %%
save_dir = get_output_path("z_range_selection")
plot_histogram_upper_slices_available(datasets, save_dir)

# %%
save_dir = get_output_path(
    "z_range_selection",
    f"normalized_profiles_offsets_{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}",
)
plot_normalized_profiles(
    datasets, timepoints=[0, 90, 180, 270], save_dir=save_dir, mode="by_position"
)

# %%
plot_normalized_profiles(
    datasets, timepoints=[0, 90, 180, 270], save_dir=save_dir, mode="by_dataset"
)
# %%
