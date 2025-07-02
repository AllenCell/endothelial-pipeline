import matplotlib.pyplot as plt
import numpy as np


def plot_crop_montage(
    list_of_crops: list[np.ndarray],
    channel_index: int = 1,  # brightfield standard dev
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a montage of 12 crops from a larger image."""

    fig, ax = plt.subplots(4, 3, figsize=(12, 16))
    for i, crop in enumerate(list_of_crops):
        if i >= 12:
            break
        ax[i // 3, i % 3].imshow(crop[channel_index, 0], cmap="gray")
    return fig, ax
