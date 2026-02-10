# %%
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import BoundaryNorm, ListedColormap

from endo_pipeline.settings.image_data import (
    IMG_SHAPE_RESOLUTION_1_3i_X,
    IMG_SHAPE_RESOLUTION_1_3i_Y,
)


def create_label_image(img_x: int, img_y: int, crop_len: int) -> np.ndarray:
    """
    Create a label image where each non-overlapping crop_len x crop_len tile
    gets a unique label 1..N in row-major order.
    Pixels outside full tiles remain 0.
    """
    n_tiles_x = (img_x - crop_len) // crop_len + 1
    n_tiles_y = (img_y - crop_len) // crop_len + 1

    label_img = np.zeros((img_y, img_x), dtype=np.int32)
    label = 1
    for row in range(n_tiles_y):
        for col in range(n_tiles_x):
            start_x = col * crop_len
            start_y = row * crop_len
            label_img[start_y : start_y + crop_len, start_x : start_x + crop_len] = label
            label += 1
    return label_img


def view_label_image(label_image: np.ndarray) -> None:
    """
    Display a label image using Glasbey colors.
    """
    labels_masked = np.ma.masked_where(label_image <= 0, label_image)
    unique_labels = np.unique(labels_masked.compressed())

    cmap = ListedColormap(cc.glasbey[: len(unique_labels)])
    norm = BoundaryNorm(np.append(unique_labels - 0.5, unique_labels[-1] + 0.5), cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(labels_masked, cmap=cmap, norm=norm, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_label_image_tiff(label_image: np.ndarray, path: str) -> None:
    """
    Save the label image as a TIFF file with uint16 dtype.
    """
    tifffile.imwrite(path, label_image.astype(np.uint16))


def add_label_to_dataframe(df: pd.DataFrame, img_x: int, img_y: int, crop_len: int) -> pd.DataFrame:
    """
    Add a 'label' column to a dataframe that contains start_x and start_y for tiles.
    Label corresponds to row-major 1..N tile numbering.
    """
    n_tiles_x = (img_x - crop_len) // crop_len + 1

    col_idx = df["start_x"] // crop_len
    row_idx = df["start_y"] // crop_len

    df["label"] = (row_idx * n_tiles_x + col_idx + 1).astype(int)
    return df


# %%
CROP = 128

# Create label image
label_img = create_label_image(IMG_SHAPE_RESOLUTION_1_3i_X, IMG_SHAPE_RESOLUTION_1_3i_Y, CROP)

# View label image
view_label_image(label_img)

# Save as TIFF uint16
save_label_image_tiff(label_img, "label_image.tif")

# Add labels to a dataframe
df = pd.DataFrame({"start_x": [0, 128, 256, 0, 128], "start_y": [0, 0, 0, 128, 128]})
df = add_label_to_dataframe(df, IMG_SHAPE_RESOLUTION_1_3i_X, IMG_SHAPE_RESOLUTION_1_3i_Y, CROP)
print(df)

# %%
