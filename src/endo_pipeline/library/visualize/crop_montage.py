import matplotlib.pyplot as plt
import pandas as pd

from src.endo_pipeline.library.visualize import viz_base


def plot_crop_montage(
    list_of_crops: list,
    df_sample_sorted: pd.DataFrame,
    pc_axis: int,
    pc_val: float,
    image_content: str,
    channel_index: int | None,
    save_dir: str,
) -> None:
    """
    Plot a montage of crops from a list of crops.

    Args:
        list_of_crops (list): List of crops to plot.
        df_sample_sorted (pd.DataFrame): DataFrame containing metadata for the crops.
        pc_axis (int): Principal component axis to use for the title.
        pc_val (float): Value of the principal component to use for the title.
        image_content (str): Content type of the image (e.g., "stddev_bf").
        channel_index (int or None): Index of the channel to plot.
            If None, the crop is assumed to be single-channel.
        save_dir (str): Directory to save the plot. If None, the plot is not saved.
    """
    fig, ax = plt.subplots(10, 10, figsize=(32, 32))
    for i, crop in enumerate(list_of_crops):
        if channel_index is None:
            # not a multichannel image
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
        ax[i // 10, i % 10].axis("off")
    fig.suptitle(f"PC{pc_axis+1} value: {pc_val}", y=1.0, fontsize=45)
    plt.tight_layout()
    plt.show()
    viz_base.save_plot(
        fig,
        save_dir
        + f"PC{pc_axis+1}_val_"
        + "p".join(str(pc_val).split("."))
        + f"_{image_content}_crops_montage",
    )
