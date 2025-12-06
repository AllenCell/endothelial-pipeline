# %%
import logging

import matplotlib.pyplot as plt

from endo_pipeline.configs import load_dataset_config, load_model_config
from endo_pipeline.io import get_output_path, load_image
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    create_data_dict_loaded_image,
    get_image_transforms,
    visualize_fov_transform_steps,
)
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_TRAIN_CONFIG
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

logger = logging.getLogger(__name__)

# %%
DESCRIPTION = "Visualize the image preprocessing steps for the DiffAE model."
TAGS = ["supfig", "preprocessing", "diffae"]

# %%
plt.style.use("endo_pipeline.figure")

# %% Load Example Data
FIGURE_ID = "SUPP_FIG_IMG_PROC"
DATASET = EXAMPLE_DATASET[FIGURE_ID]
POSITION = 0
TIMEPOINT = 0
CROP_SIZE = 128
save_dir = get_output_path("model_input_preprocessing_viz", f"{DATASET}_P{POSITION}")

dataset_config = load_dataset_config(DATASET)
zarr_loc = get_zarr_location_for_position(dataset_config, POSITION)
img = load_image(zarr_loc, level=1, timepoints=TIMEPOINT, squeeze=True, compute=True)

# %% Load model config and initialize transforms
model_config = load_model_config(DIFFAE_MODEL_TRAIN_CONFIG)
transforms = get_image_transforms(model_config)
data = create_data_dict_loaded_image(img)

# Step through each transformation and visualize the processing steps for each channel
# Panel A - BF
transformed_bf = visualize_fov_transform_steps(transforms, data, save_dir, target_key="raw_bf")
# Panel B - CDH5
transformed_cdh5 = visualize_fov_transform_steps(transforms, data, save_dir, target_key="raw_cdh5")

# %%
titles = {
    "a1": "Std. dev. Z-proj.",
    "a2": "Log transform",
    "a3": "Clip (0.01, 0.98)",
    "a4": "Z-score norm.",
    "c1": "Max int. Z-proj.",
    "c2": "Clip (0.01, 0.99)\n& rescaled -1 to 1",
}

for key, title in titles.items():
    fig, ax = plt.subplots(figsize=(2.8, 0.08))
    plt.title(title)
    plt.axis("off")
    plt.show()
    save_plot_to_path(fig, save_dir, f"{key}_title", file_format=".svg")
    plt.close(fig)

# %% Figure
x_offset = MAX_FIGURE_WIDTH / 4.2
y_offset = 1.8
x_crop_offset = 0.3
output_path = save_dir / "supp_fig_img_preprocessing.svg"
panel_specs = [
    # Titles
    {"letter": "", "file": "a1_title.svg", "x": 0.1, "y": 0, "x_crop": 0, "y_crop": 0},
    {"letter": "", "file": "a2_title.svg", "x": x_offset + 0.1, "y": 0, "x_crop": 0, "y_crop": 0},
    {
        "letter": "",
        "file": "a3_title.svg",
        "x": x_offset * 2 + 0.1,
        "y": 0,
        "x_crop": 0,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "a4_title.svg",
        "x": x_offset * 3 + 0.1,
        "y": 0,
        "x_crop": 0,
        "y_crop": 0,
    },
    # Images
    {
        "letter": "A",
        "file": "raw_bf_Projectd_scalebar50um.svg",
        "x": 0,
        "y": 0.2,
        "x_crop": x_crop_offset,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "raw_bf_LogImaged.svg",
        "x": x_offset,
        "y": 0.2,
        "x_crop": x_crop_offset,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "raw_bf_Clipd.svg",
        "x": x_offset * 2,
        "y": 0.2,
        "x_crop": x_crop_offset,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "raw_bf_NormalizeIntensityd.svg",
        "x": x_offset * 3,
        "y": 0.2,
        "x_crop": x_crop_offset,
        "y_crop": 0,
    },
    # Histograms
    {
        "letter": "B",
        "file": "raw_bf_Projectd_histogram.svg",
        "x": 0,
        "y": y_offset,
        "x_crop": 0,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "raw_bf_LogImaged_histogram.svg",
        "x": 1.9,
        "y": y_offset,
        "x_crop": 0,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "raw_bf_Clipd_histogram.svg",
        "x": 1.9 * 2,
        "y": y_offset,
        "x_crop": 0,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "raw_bf_NormalizeIntensityd_histogram.svg",
        "x": 1.9 * 3,
        "y": y_offset,
        "x_crop": 0,
        "y_crop": 0,
    },
    # CDH5 panels
    {
        "letter": "C",
        "file": "raw_cdh5_Projectd_scalebar50um.svg",
        "x": 0,
        "y": 4,
        "x_crop": x_crop_offset,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "raw_cdh5_ScaleIntensityRangePercentilesd.svg",
        "x": x_offset,
        "y": 4,
        "x_crop": x_crop_offset,
        "y_crop": 0,
    },
    {
        "letter": "D",
        "file": "raw_cdh5_Projectd_histogram.svg",
        "x": 0,
        "y": 5.8,
        "x_crop": 0,
        "y_crop": 0,
    },
    {
        "letter": "",
        "file": "raw_cdh5_ScaleIntensityRangePercentilesd_histogram.svg",
        "x": x_offset,
        "y": 5.8,
        "x_crop": 0,
        "y_crop": 0,
    },
]

panels = [
    FigurePanel(
        letter=spec["letter"],
        path=save_dir / spec["file"],
        x_position=spec["x"],
        y_position=spec["y"],
        x_offset=spec["x_crop"],
        y_offset=spec["y_crop"],
    )
    for spec in panel_specs
]
# %%
build_figure_from_panels(panels, output_path, width=MAX_FIGURE_WIDTH, height=10)

# %%
