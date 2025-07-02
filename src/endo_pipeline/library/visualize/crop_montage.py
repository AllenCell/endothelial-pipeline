import matplotlib.pyplot as plt
import numpy as np


def plot_crop_montage(list_of_crops: list[np.ndarray]) -> tuple[plt.Figure, np.ndarray]:
    """Plot a montage of 12 crops from a larger image."""

    fig, ax = plt.subplots(4, 3, figsize=(12, 16))
    for i, crop in enumerate(list_of_crops):
        if i >= 12:
            break
        ax[i // 3, i % 3].imshow(crop, cmap="gray")
    plt.tight_layout()
    plt.show()
    return fig, ax
