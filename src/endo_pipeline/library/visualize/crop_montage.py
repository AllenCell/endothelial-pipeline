import matplotlib.pyplot as plt
import numpy as np


def plot_crop_montage(
    list_of_crops: list[np.ndarray],
    channel_index: int = 1,  # brightfield standard dev
    max_num_crops: int = 100,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a montage of 12 crops from a larger image."""

    num_crops = min(len(list_of_crops), max_num_crops)

    fig, ax = plt.subplots(10, 10, figsize=(16, 16))
    for i, crop in enumerate(list_of_crops):
        if i >= num_crops:
            break
        ax[i // 10, i % 10].imshow(crop[channel_index, 0], cmap="gray")
    return fig, ax
