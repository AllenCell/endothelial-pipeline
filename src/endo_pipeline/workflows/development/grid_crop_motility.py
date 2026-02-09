# %%
import logging

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_config_dict_from_mlflow, load_image
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    apply_img_transforms,
    create_data_dict_loaded_image,
    get_image_transforms,
    get_target_image_from_sample,
)
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    get_zarr_location_for_position,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# %%
logger = logging.getLogger(__name__)

# Load model manifest and get location for run_name
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]

dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

# Model config has info about image processing steps from training
# Also has the crop size
model_config = get_config_dict_from_mlflow(model_location.mlflowid)
crop_size = model_config.model.image_shape[-1]  # assumes square crops

# %%
dataset_name = "20250618_20X"
crop_index = 0

dataframe = get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest)
dataframe_crop = dataframe[dataframe[ColumnName.CROP_INDEX] == crop_index].sort_values(
    by=ColumnName.TIMEPOINT
)
position = dataframe_crop[ColumnName.POSITION].iloc[0]
position_int = int(position[1:])
timepoint = dataframe_crop[ColumnName.TIMEPOINT].iloc[0]
crop_x_start = dataframe_crop[ColumnName.START_X].iloc[0]
crop_y_start = dataframe_crop[ColumnName.START_Y].iloc[0]
# %%

dataset_config = load_dataset_config(dataset_name)
zarr_loc = get_zarr_location_for_position(dataset_config, position_int)
img = load_image(
    zarr_loc,
    level=DIFFAE_ZARR_RESOLUTION_LEVEL,
    timepoints=[timepoint, timepoint + 1],
    squeeze=True,
    compute=True,
)
# %%
# Get zarr loading dictionary, get image processing steps
# from loaded model config (except cropping step)
# and apply the transforms for each channel
bf_crops: list[np.ndarray] = []
cdh5_crops: list[np.ndarray] = []
for i in range(img.shape[0]):
    data = create_data_dict_loaded_image(img[i])
    transforms = get_image_transforms(model_config)
    sample = apply_img_transforms(transforms, data)

    # Extract the processed conditioning and diffusion images
    # based on the output key from the transforms
    # Conditioning image can be brightfield or CDH5 depending on model,
    # but diffusion image is always CDH5 in our use case
    transformed_bf_image = get_target_image_from_sample(sample, target_key="raw_bf")
    transformed_cdh5_image = get_target_image_from_sample(sample, target_key="raw_cdh5")

    # Crop both images to the same region
    bf_crop = crop_image(transformed_bf_image, crop_x_start, crop_y_start, crop_size)
    cdh5_crop = crop_image(transformed_cdh5_image, crop_x_start, crop_y_start, crop_size)
    bf_crops.append(bf_crop)
    cdh5_crops.append(cdh5_crop)
# %%
for i in range(2):
    fig, ax = plt.subplots()
    ax.imshow(bf_crops[i].squeeze(), cmap="Grays_r")

for i in range(2):
    fig, ax = plt.subplots()
    ax.imshow(cdh5_crops[i].squeeze(), cmap="Grays_r")
# %%
bf_diff = bf_crops[1] - bf_crops[0]
fig, axs = plt.subplots()
ax.imshow(bf_diff.squeeze())
plt.show()

# %%
