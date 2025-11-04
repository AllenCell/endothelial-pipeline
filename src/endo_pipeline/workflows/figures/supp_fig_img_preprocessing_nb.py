# %%
import logging

from endo_pipeline.configs import load_dataset_config, load_model_config
from endo_pipeline.io import get_output_path, load_image
from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    create_data_dict_loaded_image,
    get_image_transforms,
    save_stack_slices_as_thumbnails,
    visualize_fov_transform_steps,
)
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_TRAIN_CONFIG
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.image_data import (
    LOWER_Z_SLICE_OFFSET,
    UPPER_Z_SLICE_OFFSET,
    PIXEL_SIZE_3i_20x,
)

logger = logging.getLogger(__name__)

# %%
DESCRIPTION = "Visualize the image preprocessing steps for the DiffAE model."
TAGS = ["supfig", "preprocessing", "diffae"]

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

# %% Panel A - Thumbnail of each slice in Z-stack for each channel
save_stack_slices_as_thumbnails(img, save_dir)

# %% Panel B - Z slices used in preprocessing steps
in_focus_slice = dataset_config.center_z_plane[POSITION]
logging.info(f"Dataset {DATASET}, position {POSITION}, in-focus z slice: {in_focus_slice}")
logging.info("Lower z slice: %s", in_focus_slice - LOWER_Z_SLICE_OFFSET)
logging.info("Upper z slice: %s", in_focus_slice + UPPER_Z_SLICE_OFFSET)

# %% Load model config and initialize transforms
model_config = load_model_config(DIFFAE_MODEL_TRAIN_CONFIG)
transforms = get_image_transforms(model_config)
data = create_data_dict_loaded_image(img)

# Step through each transformation and visualize the processing steps for each channel
# %% Panel C - BF
transformed_bf = visualize_fov_transform_steps(transforms, data, save_dir, target_key="raw_bf")
# %% Panel D - CDH5
transformed_cdh5 = visualize_fov_transform_steps(transforms, data, save_dir, target_key="raw_cdh5")


# %% Visualize cropped images
for image, name in [(transformed_bf, "crop_bf"), (transformed_cdh5, "crop_cdh5")]:
    cropped_image = crop_image(image, 100, 100, CROP_SIZE)
    plot_image_thumbnail(
        cropped_image.squeeze(),
        name,
        save_dir,
        figsize=(6, 6),
        scalebar_size_um=10,
        pixel_size=PIXEL_SIZE_3i_20x,
        bar_thickness=4,
        bar_padding=5,
    )


# %%
