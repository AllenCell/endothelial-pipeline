import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_crop_montage(
    list_of_crops: list[np.ndarray],
    df_sample_sorted: pd.DataFrame,
    pc_axis: int,
    channel_index: int | None = 1,  # brightfield standard dev
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a montage of 12 crops from a larger image."""

    fig, ax = plt.subplots(10, 10, figsize=(32, 32))
    for i, crop in enumerate(list_of_crops):
        if channel_index is None:
            # if not a multichannel image
            crop_ = crop.copy()
        else:
            # multichannel, grab one
            crop_ = crop[channel_index, 0]
        ax[i // 10, i % 10].imshow(crop_, cmap="gray")
        ax[i // 10, i % 10].set_title(
            f"{i+1}"
            # f"\n{df_sample_sorted['dataset'].iloc[i]}, "
            # f"{df_sample_sorted['frame_number'].iloc[i]}"
        )
        ax[i // 10, i % 10].axis("off")  # Turn off axis ticks after setting the title
    return fig, ax
