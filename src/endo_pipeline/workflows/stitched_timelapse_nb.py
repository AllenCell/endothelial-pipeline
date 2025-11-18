# %%
import logging

import matplotlib.pyplot as plt

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.library.process import image_processing
from endo_pipeline.library.visualize.figure_utils import add_scalebar, add_timestamp
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

DESCRIPTION = "Create stitched timelapse."

TAGS = ["visualization"]

logger = logging.getLogger(__name__)

# %%
dataset_list = get_datasets_in_collection("timelapse")
dataset_name = dataset_list[0]

dataset_config = load_dataset_config(dataset_name)
zarr_positions = dataset_config.zarr_positions

position_timelapses = []
for position in zarr_positions:
    location = get_zarr_location_for_position(dataset_config, position=position)
    image = load_image(location, channels=["EGFP"], level=0)
    image_max = image.max(axis=2).squeeze()
    position_timelapses.append(image_max)
# %%
image_stitched = image_processing.stitch_with_overlap(position_timelapses, overlap_ratio=0.10)
# %%
for i in range(image_stitched.shape[0]):
    figure, ax = plt.subplots(frameon=False)
    tp_img = image_stitched[i].squeeze().compute()
    contrasted_image = image_processing.contrast_stretching(tp_img)
    plt.imshow(contrasted_image, cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    add_scalebar(
        ax,
        scale_bar_um=100,
        pixel_size=PIXEL_SIZE_3i_20x,
        color="white",
        bar_thickness=50,
        padding=50,
    )
    add_timestamp(ax, frame=i, interval_minutes=dataset_config.time_interval_in_minutes)
    plt.show()
    break

# %%
