# %%
from typing import Sequence

import pandas as pd
from bioio import BioImage

from cellsmap.util.manifest_io import get_dataframe_by_fmsid
from src.endo_pipeline.configs import load_single_dataset_config
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import add_crop_index


# %%
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
            f"crop_index {crop_index} not found in the DataFrame. Please check the crop index."
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


# %%
# test manifest downloaded from FMS
if __name__ == "__main__":
    manifest_fmsid = "3f01f4a5ed754d06b5dbe13df19908b0"
    df_ = get_dataframe_by_fmsid(manifest_fmsid)  # get dataframe from manifest ID
    df_ = add_position(df_)  # add FOV ID
    df_ = add_crop_index(df_)  # add crop index (defined via start_x, start_y, fov_id)
