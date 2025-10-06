import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.process.get_images import (
    get_crops_in_dataframe,
    global_contrast_crop_list,
    individual_contrast_crop_list,
)

logger = logging.getLogger(__name__)


def _pc_val_to_str(pc_val: float) -> str:
    """Transform a principal component value to a string for use in filenames."""
    pc_val_string = str(pc_val)
    if pc_val < 0:
        pc_val_string = pc_val_string.replace("-", "neg")
    pc_val_string = pc_val_string.replace(".", "p")
    return pc_val_string


def _plot_crop_montage(
    list_of_crops: list,
    pc_axis: int,
    pc_val: float,
    image_content: str,
    channel_index: int | None,
    save_dir: Path,
) -> None:
    """
    Plot a montage of crops from a list of crops.

    This function is called by `generate_contact_sheet` in a loop
    to create the montages for each type of image content.

    Parameters
    ----------
    list_of_crops
        List of crops to plot.
    pc_axis
        Principal component axis to use for the title.
    pc_val
        Value of the principal component to use for the title.
    image_content
        Content type of the image (e.g., "stddev_bf").
    channel_index
        Index of the channel to plot, if applicable.
    save_dir
        Directory to save the plot.

    Returns
    -------
    :
        Plots a montage of the crops and saves it to the specified directory.
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
        ax[i // 10, i % 10].set_title(f"{i+1}")
        ax[i // 10, i % 10].axis("off")
    fig.suptitle(f"PC{pc_axis+1} value: {pc_val}", y=1.0, fontsize=45)
    plt.tight_layout()
    pc_val_str = _pc_val_to_str(pc_val)
    save_plot_to_path(
        fig,
        save_dir,
        f"PC{pc_axis+1}_val_{pc_val_str}_{image_content}_crops_montage",
    )
    plt.show()


def generate_contact_sheet(
    df_sample: pd.DataFrame,
    model_manifest_name: str,
    run_name: str | None,
    pc_axis: int,
    pc_val: float,
    fig_savedir: Path,
    num_gpus: int | None = None,
) -> None:
    """
    Generate and save montages for various image crops and various contrast enhancements.

    Parameters
    ----------
    df_sample
        DataFrame containing sampled crop metadata.
    model_manifest_name
        Name of the model manifest corresponding to the features used.
    run_name
        Name of the specific model run corresponding to the features used.
    pc_axis
        Principal component axis used for titling.
    pc_val
        Value of the principal component bin used for titling.
    fig_savedir
        Directory to save montage images.
    num_gpus
        Number of GPUs available for processing. If None, reconstruction is skipped.

    Returns
    -------
    :
        Saves montage images to the specified directory.
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
    if num_gpus is not None:
        from hydra.utils import instantiate

        from endo_pipeline.io import load_model
        from endo_pipeline.library.model.diffae.generate_image import (
            get_reconstructed_crops_in_dataframe,
        )
        from endo_pipeline.manifests import load_model_manifest

        model_manifest = load_model_manifest(model_manifest_name)
        model = load_model(model_manifest.locations[run_name])
        # have to instantiate the model specified in the configcfg
        # to get the correct object for using the generate_from_coords function
        model_correct_type = instantiate(model.cfg.model)

        reconstructed_crop_list = get_reconstructed_crops_in_dataframe(
            df_sample_sorted,
            model_correct_type,
        )
        contrast_crops["reconstructed_cdh5"] = reconstructed_crop_list
    else:
        logger.warning("GPU not available, skipping reconstruction of crops.")

    # Generate montages for each image content type
    for image_content, crop_list_channel in contrast_crops.items():
        _plot_crop_montage(
            crop_list_channel,
            pc_axis,
            pc_val,
            image_content=image_content,
            channel_index=None,
            save_dir=fig_savedir,
        )
