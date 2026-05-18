"""
Visualize the selection of Z slices for image preprocessing.

#supfig #preprocessing
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path, load_image
from endo_pipeline.library.process.z_stack_selection import (
    calculate_center_planes_all_tp_for_pos,
    plot_global_center_plane,
    plot_histogram_upper_slices_available,
    plot_standard_devs_per_slice,
    visualize_slice_selection,
)
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET

# %%
plt.style.use("endo_pipeline.figure")

# %% Load dataset
FIGURE_ID = "SUPP_FIG_Z_SLICE"
dataset = EXAMPLE_DATASET[FIGURE_ID]
save_dir = get_output_path("supp_fig_z_slice_selection")
dataset_config = load_dataset_config(dataset)
position, frame = 0, 0

# %% Load images
zarr_loc = get_zarr_location_for_position(dataset_config, position)
bf_stack = load_image(zarr_loc, channels=["BF"], timepoints=frame, level=1, squeeze=True)
cdh5_stack = load_image(zarr_loc, channels=["EGFP"], timepoints=frame, level=1, squeeze=True)

# %% Panel A - In focus Z slice selection per timepoint
stdevs = [plane.std().compute() for plane in bf_stack]
focal_plane_tp = max(0, np.argmin(stdevs))
plot_standard_devs_per_slice(
    stdevs, int(focal_plane_tp), dataset, position, frame, save_dir, (2.5, 2.15)
)

# %% Panel B - In focus Z slice selection per position over time
focal_planes = calculate_center_planes_all_tp_for_pos(dataset_config, position)
# %%
focal_plane_pos, std_dev = plot_global_center_plane(
    focal_planes,
    dataset_config.name,
    position,
    save_dir,
    (2.5, 2.15),
    show_histogram=False,
)

# %% Panel C - Distribution of upper slices available across datasets
datasets = get_datasets_in_collection("shear_stress")
plot_histogram_upper_slices_available(datasets, save_dir, figure_size=(1.5, 2.15))

# %% Panel D - Example images of selected Z slices
visualize_slice_selection(
    bf_stack,
    cdh5_stack,
    int(focal_plane_pos),
    dataset,
    position,
    frame,
    save_dir,
    (MAX_FIGURE_WIDTH * 0.75, MAX_FIGURE_WIDTH * 0.75 * 2 / 3),
    LOWER_Z_SLICE_OFFSET,
    UPPER_Z_SLICE_OFFSET,
)

# %% Figure
output_path = save_dir / "supp_fig_z_slice_selection.svg"
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / f"standard_devs_{dataset}_P{position}_{frame}.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / f"global_center_plane_{dataset}_P{position}.svg",
        x_position=2.5,
        y_position=0,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="C",
        path=save_dir / "n_slices_above_in_focus_z_histogram.svg",
        x_position=4.9,
        y_position=0,
        x_offset=0.08,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="D",
        path=save_dir
        / f"plane_selection_vis_{dataset}_P{position}_{frame}_offset{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}_scalebar100um.svg",
        x_position=0,
        y_position=2.3,
        x_offset=0.08,
        y_offset=0.08,
    ),
]
build_figure_from_panels(panels, output_path, width=MAX_FIGURE_WIDTH, height=7.2)

# %%
