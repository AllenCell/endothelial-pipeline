# %%
import matplotlib.pyplot as plt

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.library.process.image_processing import (
    contrast_stretching,
    crop_image,
    get_single_bf_plane,
    log_normalize_image,
    max_proj,
    std_dev,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.library.visualize.intro_schematic import create_intro_schematic
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.examples import FIGURE_1_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

DESCRIPTION = "Figure panels for Figure 1"

# %%
plt.style.use("endo_pipeline.figure")

# %% Panel A: Intro schematic
save_dir = get_output_path("figure_1")
fig, ax = create_intro_schematic(figure_size=(MAX_FIGURE_WIDTH, 2))
save_plot_to_path(fig, save_dir, "intro_schematic", file_format=".svg", dpi=900)

# %% Panel B: Example images from biological system at low and high shear stress
CROP_SIZE = 1000  # in pixels at res level 0

image_panel_list = []
row_titles = []
for example in FIGURE_1_EXAMPLE_IMAGES:
    dataset_config = load_dataset_config(example.dataset_name)
    shear_stress_value = int(dataset_config.flow_conditions[0].shear_stress)
    location = get_zarr_location_for_position(dataset_config, position=example.position)
    gfp_image = load_image(location, timepoints=example.timepoint, channels=["EGFP"], squeeze=True)
    bf_image = load_image(location, timepoints=example.timepoint, channels=["BF"], squeeze=True)

    gfp_max_proj = max_proj(gfp_image, axis=0)
    bf_plane = get_single_bf_plane(bf_image)
    bf_std_dev = std_dev(bf_image, axis=0)

    log_bf_std_dev = log_normalize_image(bf_std_dev)

    gfp_max_proj = contrast_stretching(gfp_max_proj)
    bf_plane = contrast_stretching(bf_plane)
    log_bf_std_dev = contrast_stretching(log_bf_std_dev)

    gfp_max_proj = crop_image(gfp_max_proj, example.crop_x_start, example.crop_y_start, CROP_SIZE)
    bf_plane = crop_image(bf_plane, example.crop_x_start, example.crop_y_start, CROP_SIZE)
    log_bf_std_dev = crop_image(
        log_bf_std_dev, example.crop_x_start, example.crop_y_start, CROP_SIZE
    )

    image_panel_list.extend([gfp_max_proj, bf_plane, log_bf_std_dev])
    row_titles.append(f"{shear_stress_value} dyn/cm²")

image_panel_fig = make_contact_sheet(
    image_panel_list,
    max_rows=2,
    max_cols=3,
    col_titles=["GFP max proj", "BF z-slice", "BF std dev proj"],
    row_titles=row_titles,
    font_size=FONTSIZE_MEDIUM,
    subplot_kwargs={"frame_on": False},
    fig_kwargs={"figsize": (MAX_FIGURE_WIDTH / 2, 2.2), "constrained_layout": True},
)

image_panel_fig.get_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

for ax in image_panel_fig.axes:
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3

scale_bar_um = 100
add_scalebar(
    image_panel_fig.axes[0],
    scale_bar_um=scale_bar_um,
    pixel_size=PIXEL_SIZE_3i_20x,
    location="lower left",
    bar_thickness=50,
    padding=50,
)

save_plot_to_path(
    image_panel_fig,
    save_dir,
    f"biological_system_examples_{shear_stress_value}_20_dyn_scale_bar_{scale_bar_um}um",
    file_format=".svg",
    tight_layout=False,
)
# %%
