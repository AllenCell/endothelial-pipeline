from pathlib import Path
from typing import Literal, Sequence

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
from bioio import BioImage
from tqdm import tqdm

from src.endo_pipeline.configs import dataset_io
from src.endo_pipeline.library.process.image_processing import (
    contrast_stretching,
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


def get_crops_in_dataframe(df: pd.DataFrame) -> tuple[list[np.ndarray], pd.DataFrame]:
    """
    Get crops of images from the dataframe for a
    given dataset and save them as multichannel TIFF files.
    Return these crops as a list of numpy arrays and a dataframe
    matching the order of the list.
    """
    # Initialize dataset name and list of images to return
    dataset = df["dataset"].iloc[0]
    image_seq = []
    sorted_rows = []  # List to store rows in the same order as images

    # Create an overall progress bar for all rows in the dataframe
    with tqdm(total=len(df), desc="Processing crops") as pbar:
        # Loop through each position in the dataframe
        for position, df_pos in df.groupby("position"):
            p = dataset_io.extract_P(position)
            img = get_zarr_img_for_dataset(dataset, p)

            # Loop through rows of the current group (rows corresponding to the current position)
            for index, row in df_pos.iterrows():
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
                bf_channel = crop[:, 1, :, :, :]  # Brightfield channel
                gfp_channel = crop[:, 0, :, :, :]  # GFP channel

                # Perform operations on the extracted channels
                bf_max_projection = max_proj(bf_channel, 1)
                bf_std_deviation = std_dev(bf_channel, 1)
                gfp_max_projection = max_proj(gfp_channel, 1)

                # Contrast stretch
                bf_max_proj_contrast = contrast_stretching(bf_max_projection, method="percentile")
                bf_std_dev_contrast = contrast_stretching(bf_std_deviation, method="percentile")
                gfp_max_proj_contrast = contrast_stretching(gfp_max_projection, method="percentile")

                # Combine the processed images into a multichannel array
                multichannel_image = np.stack(
                    [
                        bf_max_proj_contrast,
                        bf_std_dev_contrast,
                        gfp_max_proj_contrast,
                    ],
                    axis=0,  # Stack along the channel axis
                )

                image_seq.append(multichannel_image)
                sorted_rows.append(row)  # Append the row to maintain order

                # Update the overall progress bar
                pbar.update(1)

    # Create a new dataframe from the sorted rows
    df_sorted = pd.DataFrame(sorted_rows).reset_index(drop=True)

    return image_seq, df_sorted


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
