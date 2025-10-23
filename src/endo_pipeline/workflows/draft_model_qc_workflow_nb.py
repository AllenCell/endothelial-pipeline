# %%
from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config, load_model_config
from endo_pipeline.io.input import load_image_from_path
from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    apply_img_preprocessing,
)
from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_TRAIN_CONFIG

# %%
DATASET = "20250224_20X"
POSITION = 0
TIMEPOINT = 0

# %% Load Example Data
dataset_config = load_dataset_config(DATASET)
zarr_path = get_zarr_file_for_position(dataset_config, POSITION)
img = load_image_from_path(zarr_path, level=1, timepoints=TIMEPOINT, squeeze=True, compute=True)

# %% Load model config and apply preprocessing
model_config = load_model_config(DIFFAE_MODEL_TRAIN_CONFIG)
CROP_SIZE = 128  # this will get pulled from the model config in practice

bf_img_crop = apply_img_preprocessing(
    model_config, img, channel="bf", start_x=100, crop_size=CROP_SIZE
)

print(
    bf_img_crop.shape
)  # should be (1, CROP_SIZE, CROP_SIZE), I think down stream steps expect C, Y, X

cdh5_img_crop = apply_img_preprocessing(
    model_config, img, channel="cdh5", start_x=100, crop_size=CROP_SIZE
)

save_dir = get_output_path("model_input_img_preprocessing", f"{DATASET}_P{POSITION}_crops")
for img_crop, name in [(bf_img_crop, "crop_bf"), (cdh5_img_crop, "crop_cdh5")]:
    plot_image_tadd.humbnail(
        img_crop.squeeze(),
        name,
        save_dir,
        figsize=(6, 6),
    )
# %%
