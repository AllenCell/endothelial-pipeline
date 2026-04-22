# %%
"""Supplementary figure detailing computation of the drift vector fields from grid-based crop trajectories."""

import matplotlib.pyplot as plt

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_image, save_plot_to_path
from endo_pipeline.library.process.image_processing import (
    contrast_stretching,
    crop_image,
    log_normalize_image,
    std_dev,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.examples import FLOW_FIELD_CONSTRUCTION_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_HEIGHT
from endo_pipeline.settings.image_data import NATIVE_ZARR_RESOLUTION_CROP_SIZE, PIXEL_SIZE_3i_20x
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

# %%
plt.style.use("endo_pipeline.figure")

PANEL_CROP_SIZE = 2 * NATIVE_ZARR_RESOLUTION_CROP_SIZE
SCALE_BAR_UM = 20
CROP_START_X = 0
CROP_START_Y = 0

output_path = get_output_path("supp_fig_flow_field")
# %%
processed_images = []
for example in FLOW_FIELD_CONSTRUCTION_EXAMPLE_IMAGES:
    dataset_config = load_dataset_config(example.dataset_name)
    location = get_zarr_location_for_position(dataset_config, position=example.position)
    bf_image = load_image(location, timepoints=example.timepoint, channels=["BF"], squeeze=True)

    bf_std_dev = std_dev(bf_image, axis=0)

    log_bf_std_dev = log_normalize_image(bf_std_dev)
    log_bf_std_dev = contrast_stretching(log_bf_std_dev)

    log_bf_std_dev = crop_image(
        log_bf_std_dev, example.crop_x_start, example.crop_y_start, PANEL_CROP_SIZE
    )
    processed_images.append(log_bf_std_dev)

# %%
labels = ["t", "t+1"]
fig: plt.Figure = make_contact_sheet(
    processed_images,
    max_rows=len(processed_images),
    max_cols=1,
    row_titles=labels,
    fig_kwargs={"figsize": (MAX_FIGURE_HEIGHT // 4, MAX_FIGURE_HEIGHT // 2)},
)

fig.subplots_adjust(hspace=2.5)

for ax, img, label in zip(fig.axes, processed_images, labels, strict=True):
    ax.imshow(img, cmap="gray")
    ax.set_ylabel(label)
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    add_scalebar(
        ax,
        scale_bar_um=SCALE_BAR_UM,
        pixel_size=PIXEL_SIZE_3i_20x,
        location="lower right",
        bar_thickness=15,
        padding=25,
    )

    # add highlighted box to show crop region used for flow field construction
    rect = plt.Rectangle(
        (CROP_START_X, CROP_START_Y),
        NATIVE_ZARR_RESOLUTION_CROP_SIZE,
        NATIVE_ZARR_RESOLUTION_CROP_SIZE,
        edgecolor="magenta",
        facecolor="none",
        linewidth=2,
        clip_on=False,
    )
    ax.add_patch(rect)

fig.axes[-1].text(
    0.95,
    0.09,
    f"{SCALE_BAR_UM} {Unicode.MU}m",
    color="white",
    transform=fig.axes[-1].transAxes,
    fontsize=FONTSIZE_MEDIUM,
    va="bottom",
    ha="right",
)

filename = "flow_field_example_t_to_tp1"
save_plot_to_path(fig, output_path, filename, file_format=".svg")
image_panel_path = output_path / f"{filename}.svg"
# %%
