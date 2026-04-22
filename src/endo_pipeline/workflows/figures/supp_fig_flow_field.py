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
from endo_pipeline.settings.figures import FONTSIZE_XLARGE
from endo_pipeline.settings.image_data import NATIVE_ZARR_RESOLUTION_CROP_SIZE, PIXEL_SIZE_3i_20x

# %%
plt.style.use("endo_pipeline.figure")

PANEL_CROP_SIZE = 750
SCALE_BAR_UM = 100
CROP_START_X = NATIVE_ZARR_RESOLUTION_CROP_SIZE
CROP_START_Y = NATIVE_ZARR_RESOLUTION_CROP_SIZE

output_path = get_output_path("supp_fig_flow_field")
# %%
image_panel_paths = []
for example, label in zip(FLOW_FIELD_CONSTRUCTION_EXAMPLE_IMAGES, ["t", "t+1"], strict=True):
    dataset_config = load_dataset_config(example.dataset_name)
    location = get_zarr_location_for_position(dataset_config, position=example.position)
    bf_image = load_image(location, timepoints=example.timepoint, channels=["BF"], squeeze=True)

    bf_std_dev = std_dev(bf_image, axis=0)

    log_bf_std_dev = log_normalize_image(bf_std_dev)
    log_bf_std_dev = contrast_stretching(log_bf_std_dev)

    log_bf_std_dev = crop_image(
        log_bf_std_dev, example.crop_x_start, example.crop_y_start, PANEL_CROP_SIZE
    )

    fig = make_contact_sheet(
        [log_bf_std_dev], max_cols=1, max_rows=1, col_titles=[label], font_size=FONTSIZE_XLARGE
    )
    for ax in fig.axes:
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3

        add_scalebar(
            ax,
            scale_bar_um=SCALE_BAR_UM,
            pixel_size=PIXEL_SIZE_3i_20x,
            location="lower right",
            bar_thickness=25,
            padding=25,
        )

        # add highlighted box to show crop region used for flow field
        # construction
        rect = plt.Rectangle(
            (CROP_START_X, CROP_START_Y),
            NATIVE_ZARR_RESOLUTION_CROP_SIZE,
            NATIVE_ZARR_RESOLUTION_CROP_SIZE,
            edgecolor="magenta",
            facecolor="none",
            linewidth=2,
        )
        ax.add_patch(rect)

    filename = f"flow_field_example_{label.replace('+', 'p')}"
    save_plot_to_path(fig, output_path, filename, file_format=".svg")
    image_panel_paths.append(output_path / f"{filename}.svg")
# %%
