import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.endo_pipeline.io import save_plot_to_path
from src.endo_pipeline.library.process.get_images import (
    get_crops_in_dataframe,
    global_contrast_crop_list,
    individual_contrast_crop_list,
)


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
    save_plot_to_path(
        fig,
        save_dir,
        f"PC{pc_axis+1}_val_"
        + "p".join(str(pc_val).split("."))
        + f"_{image_content}_crops_montage",
    )
    plt.show()


def generate_contact_sheet(
    df_sample: pd.DataFrame,
    pc_axis: int,
    pc_val: float,
    fig_savedir: str,
) -> None:
    """
    Generate and save montages (contact sheets) for various image crops, including
    contrast-enhanced and optionally reconstructed views.

    Parameters
    ----------
    df_sample : pd.DataFrame
        DataFrame containing sampled crop metadata.
    pc_axis : int
        Principal component axis used for titling.
    pc_val : float
        Value of the principal component bin used for titling.
    fig_savedir : str
        Directory to save montage images.
    """
    # Get image crops and sorted sample DataFrame
    (
        bf_single_slice,
        bf_max_projection,
        bf_std_deviation,
        gfp_max_projection,
        df_sample_sorted,
    ) = get_crops_in_dataframe(df_sample)

    # Define raw crop types
    crop_types = {
        "bf_slice": bf_single_slice,
        "bf_max_proj": bf_max_projection,
        "stddev_bf": bf_std_deviation,
        "cdh5": gfp_max_projection,
    }

    # Generate contrast-enhanced versions
    contrast_crops = {}
    for name, crop_list in crop_types.items():
        contrast_crops[f"{name}_global_contrast"] = global_contrast_crop_list(
            crop_list, "percentile"
        )
        contrast_crops[f"{name}_ind_contrast"] = individual_contrast_crop_list(
            crop_list, "percentile"
        )

    # Optionally add reconstructed crops (if GPU is available)
    if torch.cuda.is_available():
        from src.endo_pipeline.library.model.diffae.generate_image import (
            get_reconstructed_crops_in_dataframe,
        )

        reconstructed_crop_list = get_reconstructed_crops_in_dataframe(df_sample_sorted)
        contrast_crops["reconstructed_cdh5"] = reconstructed_crop_list
    else:
        print("GPU not available, skipping reconstruction of crops.")

    # Generate montages
    for image_content, crop_list_channel in contrast_crops.items():
        plot_crop_montage(
            crop_list_channel,
            df_sample_sorted,
            pc_axis,
            pc_val,
            image_content=image_content,
            channel_index=None,
            save_dir=fig_savedir,
        )
