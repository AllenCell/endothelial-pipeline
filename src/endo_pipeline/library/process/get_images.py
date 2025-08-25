from pathlib import Path
from typing import Literal, cast

import dask.array as da
import numpy as np
import pandas as pd
from bioio import BioImage
from tqdm import tqdm

from endo_pipeline.configs import dataset_io
from endo_pipeline.library.process.image_processing import (
    contrast_stretching,
    get_global_custom_range,
    get_single_bf_plane,
    max_proj,
    std_dev,
)


def get_zarr_img_for_dataset(
    dataset: str, position: int, resolution_level: Literal[0, 1] = 1
) -> BioImage:
    """Retrieve the BioImage object for a given dataset and position."""
    zarr_name = dataset_io.get_zarr_name(dataset, position)
    zarr_path = dataset_io.get_zarr_dir(dataset)
    filepath = Path(zarr_path) / zarr_name
    img = BioImage(filepath)
    img.set_resolution_level(resolution_level)
    return img


def _get_crop(
    img: BioImage,
    channel: int | list | None,
    timepoint: int | None,
    start_x: int,
    start_y: int,
    crop_size_x: int,
    crop_size_y: int,
) -> da.Array:
    img_crop = img.get_image_dask_data(
        "TCZYX",
        T=timepoint,
        channel=channel,
        Y=slice(start_y, start_y + crop_size_y),
        X=slice(start_x, start_x + crop_size_x),
    )
    return img_crop


def get_crops_in_dataframe(df: pd.DataFrame) -> tuple[
    list[np.ndarray],  # Brightfield single slice crops
    list[np.ndarray],  # Brightfield max projection crops
    list[np.ndarray],  # Brightfield standard deviation crops
    list[np.ndarray],  # GFP max projection crops
    pd.DataFrame,  # Sorted dataframe
]:
    """
    Get crops of images from a dataframe with metadata for those crops
    and save in individual lists for each channel/projection.

    Dataframe should contain the following columns:
        - dataset: Name of the dataset
        - position: Position of the image
        - frame_number: Timepoint of the image
        - start_x: Starting x-coordinate of the crop
        - start_y: Starting y-coordinate of the crop
        - crop_size_x: Width of the crop
        - crop_size_y: Height of the crop

    Parameters
    ----------
    df
        DataFrame containing crop metadata.

    Returns
    -------
    :
        List of crops of a single slice for brightfield images.
    :
        List of crops of max projection for brightfield images.
    :
        List of crops of standard deviation for brightfield images.
    :
        List of crops of max projection for GFP images.
    :
        DataFrame sorted in the same order as images.
    """
    crops_bf_single_slice = []  # List to store brightfield single slice crops
    crops_bf_max_projection = []  # List to store brightfield max projection crops
    crops_bf_std_deviation = []  # List to store brightfield standard deviation crops
    crops_gfp_max_projection = []  # List to store GFP max projection crops
    sorted_rows = []  # List to store rows in the same order as images

    # loop through each dataset in the dataframe
    for dataset, df_dataset in df.groupby("dataset"):
        dataset = cast(str, dataset)  # Ensure dataset is a string
        with tqdm(total=len(df_dataset), desc=f"Processing crops for {dataset}") as pbar:
            # Loop through each position available in the dataset
            for position, df_pos in df_dataset.groupby("position"):
                position = cast(str, position)  # Ensure position is a string
                position_integer = int(position[-1])  # Extract the position number from the string
                img = get_zarr_img_for_dataset(dataset, position_integer, resolution_level=1)
                for _, row in df_pos.iterrows():
                    timepoint = row["frame_number"]
                    crop = _get_crop(
                        img,
                        channel=None,
                        timepoint=timepoint,
                        start_x=row["start_x"],
                        start_y=row["start_y"],
                        crop_size_x=row["crop_size_x"],
                        crop_size_y=row["crop_size_y"],
                    )

                    # Extract channels once, these channel indecies are hardcoded
                    # because we defined the order of channels in the zarr
                    bf_channel = crop[:, 1, :, :, :].squeeze()
                    gfp_channel = crop[:, 0, :, :, :].squeeze()

                    # Process channels
                    crops_bf_single_slice.append(get_single_bf_plane(bf_channel))
                    crops_bf_max_projection.append(max_proj(bf_channel, 0))
                    crops_bf_std_deviation.append(std_dev(bf_channel, 0))
                    crops_gfp_max_projection.append(max_proj(gfp_channel, 0))

                    sorted_rows.append(row)
                    pbar.update(1)

    # return dataframe with the same order as images
    df_sorted = pd.DataFrame(sorted_rows).reset_index(drop=True)

    return (
        crops_bf_single_slice,
        crops_bf_max_projection,
        crops_bf_std_deviation,
        crops_gfp_max_projection,
        df_sorted,
    )


def global_contrast_crop_list(
    crop_list: list,
    contrast_method: Literal["min-max", "percentile"] = "percentile",
) -> list[np.ndarray]:
    """
    Apply the same contrast stretching to all crops in the list.

    Parameters
    ----------
    crop_list
        List of crops to apply contrast stretching to.
    channel_index
        Index of the channel to apply contrast stretching on.
    contrast_method
        Method for contrast stretching.

    Returns
    -------
    :
        List of crops with contrast stretching applied.
    """
    low, high = get_global_custom_range(crop_list, method=contrast_method)

    contrasted_channel = []
    for crop in crop_list:
        contrast_crop = contrast_stretching(crop, custom_range=(low, high))
        contrasted_channel.append(contrast_crop)
    return contrasted_channel


def individual_contrast_crop_list(
    crop_list: list,
    contrast_method: Literal["min-max", "percentile"] = "percentile",
) -> list[np.ndarray]:
    """
    Apply individual contrast stretching to each crop in the list.

    Parameters
    ----------
    crop_list
        List of crops to apply contrast stretching to.
    contrast_method
        Method for contrast stretching.

    Returns
    -------
    :
        List of crops with contrast stretching applied.
    """
    contrasted_channel = []
    for crop in crop_list:
        contrast_crop = contrast_stretching(crop, method=contrast_method)
        contrasted_channel.append(contrast_crop)
    return contrasted_channel
