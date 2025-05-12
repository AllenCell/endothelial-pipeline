# %%
import os

import matplotlib.pyplot as plt
import tifffile
from cellpose import models
from skimage.color import label2rgb

from cellsmap.util import dataset_io
from cellsmap.vis import get_images, image_processing

# %%
OUTPUT_DIR = (
    f"//allen/aics/endothelial/morphological_features/segmentations/nuclear_stain_seg/"
)
DATASET = "20250122_SMAD1"
# %%
info = dataset_io.get_dataset_info(DATASET)
n_positions = dataset_io.get_total_number_of_positions(DATASET)

max_int_projections: list
max_int_projections = []
for position in range(n_positions):
    print(f"Position {position}")

    img = get_images.get_zarr_img_for_dataset(DATASET, position, resolution_level=0)
    img_tp = img.get_image_dask_data("ZYX", T=0, C=2)
    max_int_projection = image_processing.max_proj(img_tp, axis=0)
    max_int_projections.append(max_int_projection)

# %%
model = models.Cellpose(model_type="nuclei")

masks, flows, styles, diams = model.eval(
    max_int_projections, diameter=None, channels=[0, 0]
)

# %%
for max_int_proj, mask in zip(max_int_projections, masks):

    colored_mask = label2rgb(mask, bg_label=0, kind="overlay")

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(max_int_proj, cmap="gray")
    axes[0].set_title("Original DAPI Image")
    axes[0].axis("off")

    axes[1].imshow(colored_mask)
    axes[1].set_title("Nuclear Segmentation Mask")
    axes[1].axis("off")

    axes[2].imshow(max_int_proj, cmap="gray")  # Show original DAPI first
    axes[2].imshow(
        colored_mask, alpha=0.2
    )  # Add the colored mask with transparency overlay
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    print(f"Raw shape: {max_int_proj.shape}, Mask shape: {mask.shape}")

    plt.tight_layout()
    plt.show()
# %%
for mask, position in zip(masks, range(n_positions)):
    save_path = f"{OUTPUT_DIR}/{DATASET}/P{position}/"
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    tifffile.imwrite(save_path + f"{DATASET}_P{position}_T0.ome.tiff", mask)

# %%
