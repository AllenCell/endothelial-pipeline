import matplotlib.pyplot as plt
import numpy as np


def plot_crop_montage(
    list_of_crops: list[np.ndarray],
    channel_index: int | None = 1,  # brightfield standard dev
    max_num_crops: int = 100,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a montage of 12 crops from a larger image."""

    # only plot up to max_num_crops crops
    num_crops = min(len(list_of_crops), max_num_crops)
    list_of_crops_ = list_of_crops[:num_crops]

    fig, ax = plt.subplots(10, 10, figsize=(32, 32))
    for i, crop in enumerate(list_of_crops_):
        if channel_index is None:
            # if not a multichannel image
            crop_ = crop.copy()
        else:
            # multichannel, grab one
            crop_ = crop[channel_index, 0]
        ax[i // 10, i % 10].imshow(crop_, cmap="gray")
    return fig, ax
