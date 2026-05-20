"""
Visualize the image preprocessing steps for the DiffAE model.

#supfig #preprocessing #diffae
"""

# %%
import logging

import matplotlib.pyplot as plt

from endo_pipeline.configs import load_dataset_config, load_model_config
from endo_pipeline.io import get_output_path, load_image
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

# %% Step through each transformation and visualize the processing steps for each channel
PANEL_SIZE_BF = MAX_FIGURE_WIDTH * (2 / 3)
transformed_bf = visualize_fov_transform_steps(
    transforms,
    data,
    save_dir,
    target_key="raw_bf",
    figure_size=(PANEL_SIZE_BF, 1.5),
    col_titles=["Std. dev. Z-proj.", "Log norm.", "Clip (0.1, 0.98)", "Z-score norm."],
    row_title="BF",
)

# %%
PANEL_SIZE_CDH5 = MAX_FIGURE_WIDTH * (1 / 3)
transformed_cdh5 = visualize_fov_transform_steps(
    transforms,
    data,
    save_dir,
    target_key="raw_cdh5",
    figure_size=(PANEL_SIZE_CDH5, 1.5),
    col_titles=["MIP", "Clip (0.1, 0.98), Rescale"],
    row_title="VE-cadherin",
)

# %% Figure
output_path = save_dir / "supp_fig_img_preprocessing.svg"
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "raw_bf_images.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / "raw_cdh5_images.svg",
        x_position=PANEL_SIZE_BF,
        y_position=0,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="C",
        path=save_dir / "raw_bf_histograms.svg",
        x_position=0,
        y_position=1.4,
        x_offset=0,
        y_offset=0,
    ),
    FigurePanel(
        letter="D",
        path=save_dir / "raw_cdh5_histograms.svg",
        x_position=PANEL_SIZE_BF,
        y_position=1.4,
        x_offset=0,
        y_offset=0,
    ),
]

build_figure_from_panels(panels, output_path, width=MAX_FIGURE_WIDTH, height=3)

# %%
