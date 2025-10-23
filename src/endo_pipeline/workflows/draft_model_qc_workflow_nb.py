# %%
from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config, load_model_config
from endo_pipeline.io.input import load_image_from_path
from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    apply_img_transforms,
    create_data_dict_loaded_image,
    get_image_transforms,
    get_target_image_from_sample,
)
from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_TRAIN_CONFIG

# %%
DATASET = "20250224_20X"
POSITION = 0
TIMEPOINT = 0
START_X = 100
START_Y = 100

# %% Load Example Data
dataset_config = load_dataset_config(DATASET)
zarr_path = get_zarr_file_for_position(dataset_config, POSITION)
img = load_image_from_path(zarr_path, level=1, timepoints=TIMEPOINT, squeeze=True, compute=True)

# %% Load model config and apply preprocessing
model_config = load_model_config(DIFFAE_MODEL_TRAIN_CONFIG)
CROP_SIZE = 128  # this will get pulled from the model config in practice

# %%
data = create_data_dict_loaded_image(img)
transforms = get_image_transforms(model_config)
sample = apply_img_transforms(transforms, data)

transformed_bf = get_target_image_from_sample(sample, target_key=f"raw_bf")
transformed_cdh5 = get_target_image_from_sample(sample, target_key=f"raw_cdh5")

bf_crop = crop_image(transformed_bf, START_X, START_Y, CROP_SIZE)
cdh5_img_crop = crop_image(transformed_cdh5, START_X, START_Y, CROP_SIZE)


print(
    bf_crop.shape
)  # should be (1, CROP_SIZE, CROP_SIZE), I think down stream steps expect C, Y, X

save_dir = get_output_path("model_input_img_preprocessing", f"{DATASET}_P{POSITION}_crops")
for img_crop, name in [(bf_crop, "crop_bf"), (cdh5_img_crop, "crop_cdh5")]:
    plot_image_thumbnail(
        img_crop.squeeze(),
        name,
        save_dir,
        figsize=(6, 6),
    )
# %%
