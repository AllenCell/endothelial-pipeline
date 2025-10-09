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
    calculate_center_planes_all_tp_for_pos,
    plot_global_center_plane,
    plot_histogram_upper_slices_available,
    plot_standard_devs_per_slice,
    visualize_slice_selection,
)
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.image_data import LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET

# %%
DESCRIPTION = "Visualize the selection of Z slices for image preprocessing."
TAGS = ["supfig", "preprocessing"]

# %% Load dataset
FIGURE_ID = "SUPP_FIG_Z_SLICE"
dataset = EXAMPLE_DATASET[FIGURE_ID]
save_dir = get_output_path("supp_fig_z_slice_selection")
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
focal_plane_tp = max(0, np.argmin(stdevs))
plot_standard_devs_per_slice(stdevs, int(focal_plane_tp), dataset, position, frame, save_dir)


# %% Panel B - In focus Z slice selection per position over time
focal_planes = calculate_center_planes_all_tp_for_pos(dataset_config, position)
focal_plane_pos, std_dev = plot_global_center_plane(
    focal_planes, dataset_config.name, position, save_dir, show_histogram=False
)

# %% Panel C - Distribution of upper slices available across datasets
datasets = get_datasets_in_collection("timelapse")
plot_histogram_upper_slices_available(datasets, save_dir)


# %% Panel D - Example images of selected Z slices
visualize_slice_selection(
    bf_stack,
    cdh5_stack,
    int(focal_plane_pos),
    dataset,
    position,
    frame,
    save_dir,
    LOWER_Z_SLICE_OFFSET,
    UPPER_Z_SLICE_OFFSET,
)
