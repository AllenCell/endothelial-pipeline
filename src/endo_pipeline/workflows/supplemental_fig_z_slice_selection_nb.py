# %%
import numpy as np

from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from endo_pipeline.io import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.process.z_stack_selection import (
    plot_global_center_plane,
    plot_histogram_upper_slices_available,
    plot_normalized_profiles,
    plot_standard_devs_per_slice,
    visualize_slice_selection,
)
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.image_data import LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET

# %% Load dataset
dataset = EXAMPLE_DATASET["SUP_FIG_Z_SLICE"]
save_dir = get_output_path(__file__, dataset)
dataset_config = load_dataset_config(dataset)
position, frame = 0, 0


# %% Load images
zarr_file = get_zarr_file_for_position(dataset_config, position)
bf_stack = load_zarr_as_dask_array(zarr_file, channels=["BF"], timepoints=frame, level=1).squeeze()
cdh5_stack = load_zarr_as_dask_array(
    zarr_file, channels=["EGFP"], timepoints=frame, level=1
).squeeze()


# %% Panel A - In focus Z slice selection per timepoint
stdevs = [plane.std().compute() for plane in bf_stack]
center_plane = max(0, np.argmin(stdevs))
plot_standard_devs_per_slice(stdevs, center_plane, dataset, position, frame, save_dir)


# %% Panel B - In focus Z slice selection per position over time
bf_stack_all_frames = load_zarr_as_dask_array(zarr_file, channels=["BF"], level=1)
center_planes = []

for frame in range(0, dataset_config.duration, 1):
    # Extract the BF stack for the current frame
    bf_stack = bf_stack_all_frames[frame].squeeze()

    # Compute standard deviations for all slices in the current frame
    stdevs = bf_stack.std(axis=(1, 2)).compute()

    # Find the center plane with the minimum standard deviation
    center_plane = max(0, np.argmin(stdevs))
    center_planes.append(center_plane)

mean, std_dev = plot_global_center_plane(
    center_planes, dataset_config.name, position, save_dir, show_histogram=False
)

# %% Panel C - Distribution of upper slices available across datasets
save_dir = get_output_path(__file__)
datasets = get_datasets_in_collection("timelapse")
plot_histogram_upper_slices_available(datasets, save_dir)


# %% Panel D - Example images of selected Z slices
visualize_slice_selection(
    bf_stack,
    cdh5_stack,
    center_plane,
    dataset,
    position,
    frame,
    save_dir,
    LOWER_Z_SLICE_OFFSET,
    UPPER_Z_SLICE_OFFSET,
)

# %% TBD
datasets = get_datasets_in_collection("timelapse")
save_dir = get_output_path(
    "z_range_selection",
    f"normalized_profiles_offsets_{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}",
)
plot_normalized_profiles(
    datasets, timepoints=[0], save_dir=save_dir, mode="by_position", n_positions=0
)
