# %%
import logging

import matplotlib.pyplot as plt
import numpy as np
from skimage.registration import optical_flow_tvl1

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
dataset_name = "20250319_20X"
position_int = 1
position = f"P{position_int}"
row_num = 0  # has to be between 0 and 6
col_num = 4  # has to be between 0 and 6
crop_x_start = row_num * crop_size
crop_y_start = col_num * crop_size

timepoint = 287

dataframe = get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest)
dataframe_crop = dataframe[
    (dataframe[ColumnName.POSITION] == position)
    & (dataframe[ColumnName.START_X] == crop_x_start)
    & (dataframe[ColumnName.START_Y] == crop_y_start)
].sort_values(by=ColumnName.TIMEPOINT)

if timepoint not in dataframe_crop[ColumnName.TIMEPOINT].values:
    timepoint_new = dataframe_crop[ColumnName.TIMEPOINT].iloc[
        (dataframe_crop[ColumnName.TIMEPOINT] - timepoint).abs().argsort()[0]
    ]
    logger.warning(
        "Timepoint [ %s ] not in dataframe for dataset [ %s ], position [ %s ], using next closest timepoint instead: [ %s ]",
        timepoint,
        dataset_name,
        position,
        timepoint_new,
    )
    timepoint = timepoint_new
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
v, u = optical_flow_tvl1(bf_crops[0].squeeze(), bf_crops[1].squeeze())

# vector orientation relative to (1, 0): arctan(v/u)
vec_orientation = np.arctan2(v, u)
fig, ax = plt.subplots()
im = ax.imshow(vec_orientation, cmap="hsv")
# add colorbar with ticks at -pi, -pi/2, 0, pi/2, pi and labels
cbar = plt.colorbar(im, ax=ax, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
cbar.ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

fig, ax = plt.subplots()
hist, bins = np.histogram(vec_orientation.flatten(), bins=50)
# find argmax of histogram and corresponding bin center
argmax_hist_idx = np.argmax(hist)
bin_centers = (bins[:-1] + bins[1:]) / 2
argmax_hist = bin_centers[argmax_hist_idx]
print(f"Argmax of histogram: {argmax_hist:.2f} radians")
ax.hist(vec_orientation.flatten(), bins=50)
# set x-ticks to be at -pi, -pi/2, 0, pi/2, pi and labels
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])

fig, ax = plt.subplots()
ax.quiver(u, v)
ax.set_aspect("equal")
# %%
