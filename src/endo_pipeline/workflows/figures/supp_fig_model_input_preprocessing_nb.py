# %%
import logging

from omegaconf import OmegaConf

from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
from endo_pipeline.io.input import load_model_config_from_path, load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path, save_thumbnail_to_path
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    initialize_transform,
    run_and_visualize_transforms,
)
from endo_pipeline.settings import RELATIVE_PATH_TO_TRAIN_CONFIG
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
DATASET = EXAMPLE_DATASET["SUPP_FIG_IMG_PROC"]
POSITION = 0
save_dir = get_output_path("model_input_preprocessing_viz", f"{DATASET}_P{POSITION}")

dataset_config = load_dataset_config(DATASET)
zarr_path = get_zarr_file_for_position(dataset_config, POSITION)
img = load_zarr_as_dask_array(zarr_path, level=1, squeeze=True)

# %% Panel A - Thumbnail of each slice in Z-stack for each channel
for channel in range(img.shape[0]):
    for slice_idx in range(img.shape[1]):
        slice_img = img[channel, slice_idx, :, :].compute()
        save_thumbnail_to_path(
            slice_img,
            f"original_channel{channel}_slice{slice_idx}",
            save_dir,
            figsize=(6, 6),
            scalebar_size_um=50,
            pixel_size=PIXEL_SIZE_3i_20x,
        )
# %% Panel B - Z slices used in preprocessing steps
in_focus_slice = dataset_config.center_z_plane[POSITION]
logging.info(f"Dataset {DATASET}, position {POSITION}, in-focus z slice: {in_focus_slice}")
logging.info("Lower z slice:", in_focus_slice - LOWER_Z_SLICE_OFFSET)
logging.info("Upper z slice:", in_focus_slice + UPPER_Z_SLICE_OFFSET)

# %% Load model config
model_config = load_model_config_from_path(RELATIVE_PATH_TO_TRAIN_CONFIG)

# Access the training transform configuration
train_transform_cfg = model_config.data.train_dataloaders.dataset.transform
temp_squeeze = {"_target_": "monai.transforms.SqueezeDimd", "keys": "raw", "dim": 0}
# Insert squeeze into the list of transforms so that we can visualize the 2D image
train_transform_cfg.transforms.insert(1, OmegaConf.create(temp_squeeze))

sample = {"original_path": str(zarr_path)}

# Initialize all transforms in the pipeline
transforms = [initialize_transform(t) for t in train_transform_cfg.transforms]


# Step through each transformation and visualize the output
# %%
run_and_visualize_transforms(transforms, sample, save_dir, target_key="raw_bf")
# %%
run_and_visualize_transforms(transforms, sample, save_dir, target_key="raw_cdh5")
# %%
