from pathlib import Path
from typing import Literal, Sequence

import dask.array as da
import numpy as np
import pandas as pd
from bioio import BioImage
from tqdm import tqdm

from src.endo_pipeline.configs import dataset_io
from src.endo_pipeline.library.process.image_processing import (
    contrast_stretching,
    get_global_custom_range,
    get_single_bf_plane,
    max_proj,
    std_dev,
)


def get_zarr_img_for_dataset(
    dataset: str, position: int, resolution_level: Literal[0, 1] = 1
) -> BioImage:
    """
    Retrieve the BioImage object for a given dataset and position.
    """
    zarr_name = dataset_io.get_zarr_name(dataset, position)
    zarr_path = dataset_io.get_zarr_dir(dataset)
    filepath = Path(zarr_path) / zarr_name
    img = BioImage(filepath)
    img.set_resolution_level(resolution_level)
    return img


def get_crop(
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


def get_crops_in_dataframe(df: pd.DataFrame, contrast_crops_individually: bool = False) -> tuple[
    list[np.ndarray],  # Brightfield single slice crops
    list[np.ndarray],  # Brightfield max projection crops
    list[np.ndarray],  # Brightfield standard deviation crops
    list[np.ndarray],  # GFP max projection crops
    pd.DataFrame,  # Sorted dataframe
]:
    """
    Get crops of images from the dataframe for a
    given dataset and save in individual lists for each channel.
    """
    dataset = df["dataset"].iloc[0]

    crops_bf_single_slice = []  # List to store brightfield single slice crops
    crops_bf_max_projection = []  # List to store brightfield max projection crops
    crops_bf_std_deviation = []  # List to store brightfield standard deviation crops
    crops_gfp_max_projection = []  # List to store GFP max projection crops
    sorted_rows = []  # List to store rows in the same order as images

    with tqdm(total=len(df), desc="Processing crops") as pbar:
        # Loop through each position in the dataframe
        for position, df_pos in df.groupby("position"):
            p = dataset_io.extract_P(position)
            img = get_zarr_img_for_dataset(dataset, p)

            for _, row in df_pos.iterrows():
                timepoint = row["frame_number"]
                crop = get_crop(
                    img,
                    channel=None,
                    timepoint=timepoint,
                    start_x=row["start_x"],
                    start_y=row["start_y"],
                    crop_size_x=row["crop_size_x"],
                    crop_size_y=row["crop_size_y"],
                )

                # Extract channels once
                bf_channel = crop[:, 1, :, :, :]
                gfp_channel = crop[:, 0, :, :, :]

                bf_single_slice = get_single_bf_plane(bf_channel.squeeze())
                bf_max_projection = max_proj(bf_channel.squeeze(), 0)
                bf_std_deviation = std_dev(bf_channel.squeeze(), 0)
                gfp_max_projection = max_proj(gfp_channel.squeeze(), 0)

                if contrast_crops_individually:
                    bf_single_slice = contrast_stretching(bf_single_slice, "percentile")
                    bf_max_projection = contrast_stretching(bf_max_projection, "percentile")
                    bf_std_deviation = contrast_stretching(bf_std_deviation, "percentile")
                    gfp_max_projection = contrast_stretching(gfp_max_projection, "percentile")

                crops_bf_single_slice.append(bf_single_slice)
                crops_bf_max_projection.append(bf_max_projection)
                crops_bf_std_deviation.append(bf_std_deviation)
                crops_gfp_max_projection.append(gfp_max_projection)
                sorted_rows.append(row)
                pbar.update(1)

    df_sorted = pd.DataFrame(sorted_rows).reset_index(drop=True)

    return (
        crops_bf_single_slice,
        crops_bf_max_projection,
        crops_bf_std_deviation,
        crops_gfp_max_projection,
        df_sorted,
    )


def get_channel_from_list(crop_list: list, channel_index: int) -> list[np.ndarray]:
    """
    Extract a specific channel from a list of crops and reduce the shape to (Y, X).

    Args:
        crop_list (list): List of crops, each crop is a numpy array with shape (C, Z, Y, X).
        channel_index (int): Index of the channel to extract.

    Returns:
        list[np.ndarray]: List of extracted channels with shape (Y, X).
    """
    return [crop[channel_index] for crop in crop_list]


def global_contrast_crop_list_channel(
    crop_list: list,
    # channel_index: int,
    contrast_method: Literal["min-max", "percentile"] = "percentile",
) -> list[np.ndarray]:
    """
    Apply the same contrast stretching to all crops in the list.

    Args:
        crop_list (list): List of crops to apply contrast stretching to.
        channel_index (int): Index of the channel to apply contrast stretching on.
        contrast_method (str): Method for contrast stretching.

    Returns:
        contrasted_crops (list): List of crops with contrast stretching applied.
    """
    # channel = get_channel_from_list(crop_list, channel_index)
    low, high = get_global_custom_range(crop_list, method=contrast_method)

    contrasted_channel = []
    for crop in crop_list:
        contrast_crop = contrast_stretching(crop, custom_range=(low, high))
        contrasted_channel.append(contrast_crop)
    return contrasted_channel


def add_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract position from zarr path and add it as
    its own column to the dataframe.

    This is needed for the current test manifest downloaded from FMS.
    """
    df["position"] = df.zarr_path.apply(lambda s: s.split("/")[-1].split("_")[-1].split(".")[0])
    return df


def load_egfp_crop_image(row: pd.Series) -> BioImage:
    """
    Load the VE-Cad EGFP maximum projection image
    for a given crop from the zarr path.
    The input row should be a single row of the dataframe
    containing the zarr path, crop coordinates, and time point
    (frame in the movie).
    """
    # load image from zarr path
    img = BioImage(row.zarr_path)
    img.set_resolution_level(1)
    img = img.get_image_dask_data("ZYX", C=0, T=row["frame_number"]).max(0)
    # crops are 128x128, hardcoded for now
    img = img[row.start_y : row.start_y + 128, row.start_x : row.start_x + 128]
    return img


def get_images_for_crop(
    df: pd.DataFrame, crop_index: str, frame_range: Sequence | None = None
) -> list[BioImage]:
    """
    For a given crop index, load the corresponding VE-Cadherin max
    projection images for the specified time points (frames).
    The input DataFrame should be a single dataset that has columns
    with the zarr path, crop index, and timepoint frame_number.
    The crop index is a unique identifier for each crop in the dataset
    based on the start_x, start_y coordinates and position.
    """
    try:
        df_crop_location = df.loc[df["crop_index"] == crop_index]
    except KeyError:
        raise KeyError(
            f"crop_index {crop_index} not found in the DataFrame." + "Please check the crop index."
        )

    df_crop_location.sort_values(
        by=["frame_number"], inplace=True
    )  # sort by timepoint so images are in order

    if frame_range is None:  # default to all timepoints
        frame_range = df_crop_location["frame_number"]

    # for rows in df_crop_location, load the image
    images = []
    for frame in frame_range:
        row = df_crop_location.iloc[frame]
        img = load_egfp_crop_image(row)
        images.append(img)
    return images
