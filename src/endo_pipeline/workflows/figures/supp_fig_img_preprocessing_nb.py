# %%
import logging

from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config, load_model_config
from endo_pipeline.io.input import load_image_from_path
from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    prepare_train_transforms,
    run_and_visualize_transforms,
    save_stack_slices_as_thumbnails,
)
from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_TRAIN_CONFIG
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.image_data import LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET

logger = logging.getLogger(__name__)

# %%
DESCRIPTION = "Visualize the image preprocessing steps for the DiffAE model."
TAGS = ["supfig", "preprocessing", "diffae"]

# %% Load Example Data
FIGURE_ID = "SUPP_FIG_IMG_PROC"
DATASET = EXAMPLE_DATASET[FIGURE_ID]
POSITION = 0
save_dir = get_output_path("model_input_preprocessing_viz", f"{DATASET}_P{POSITION}")

dataset_config = load_dataset_config(DATASET)
zarr_path = get_zarr_file_for_position(dataset_config, POSITION)
img = load_image_from_path(zarr_path, level=1, squeeze=True)

# %% Panel A - Thumbnail of each slice in Z-stack for each channel
save_stack_slices_as_thumbnails(img, save_dir)

# %% Panel B - Z slices used in preprocessing steps
in_focus_slice = dataset_config.center_z_plane[POSITION]
logging.info(f"Dataset {DATASET}, position {POSITION}, in-focus z slice: {in_focus_slice}")
logging.info("Lower z slice: %s", in_focus_slice - LOWER_Z_SLICE_OFFSET)
logging.info("Upper z slice: %s", in_focus_slice + UPPER_Z_SLICE_OFFSET)

# %% Load model config and initialize transforms
sample = {"original_path": str(zarr_path)}
model_config = load_model_config(DIFFAE_MODEL_TRAIN_CONFIG)
transforms = prepare_train_transforms(model_config)

# Step through each transformation and visualize the processing steps for each channel
# %% Panel C - BF
run_and_visualize_transforms(transforms, sample, save_dir, target_key="raw_bf")
# %% Panel D - CDH5
run_and_visualize_transforms(transforms, sample, save_dir, target_key="raw_cdh5")
# %%
