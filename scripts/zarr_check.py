# %%
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from bioio import BioImage

from cellsmap.util import dataset_io


def get_channel_crop(
    img: BioImage, T: int, C: int, crop_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """Helper function to get cropped data for a specific channel.

    Args:
        img (BioImage): The BioImage object containing the image data.
        T (int): The timepoint index.
        C (int): The channel index.
        crop_size (Tuple[int, int], optional): The crop size (height, width). Defaults to (128, 128).

    Returns:
        np.ndarray: The cropped image data as a NumPy array.
    """
    return img.get_image_dask_data("ZYX", T=T, C=C)[
        :,  # Keep all Z-slices
        0 : crop_size[0],  # Crop along Y-axis
        0 : crop_size[1],  # Crop along X-axis
    ]


# %%
#  Quickly visualize crop in first position, first timepoint of each zarr to confirm channel order is correct
for dataset_name in dataset_io.get_available_datasets():
    print(f"Dataset: {dataset_name}")
    zarr_paths = dataset_io.get_zarr_path(dataset_name)
    for name, position_path in zarr_paths.items():
        img = BioImage(position_path)
        print(f"Image shape: {img.shape}")
        n_channels = img.shape[1]

        # Compute projections for all channels
        channel_projections = []
        for c in range(n_channels):
            channel = get_channel_crop(img, T=0, C=c)
            if c == 1:  # Special case for Channel 1 (BF): use center slice
                projection = channel[channel.shape[0] // 2, :, :]
            else:  # Default: use max projection
                projection = channel.max(axis=0)
            channel_projections.append(projection)

        # Plot all channels
        fig, axes = plt.subplots(1, n_channels, figsize=(6 * n_channels, 6))
        if n_channels == 1:
            axes = [axes]  # Ensure axes is iterable for a single channel
        for c, ax in enumerate(axes):
            ax.imshow(channel_projections[c], cmap="gray")
            ax.set_title(f"{dataset_name} - Channel {c}")
        plt.show()

# %%
