# %%
from cellsmap.analyses.immunofloursecence.if_feature_extraction import (
    get_raw_intensity_crop,
    get_segmentation_mask_crop,
    background_subtract,
    total_intensity,
    sum_projection,
    sum_in_mask,
    sum_not_in_mask,
)
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from typing import Tuple


def process_channel(
    row: pd.Series, channel: int, seg_mask: np.ndarray, camera_offset: int = 100
) -> Tuple[float, float, float]:
    """
    Processes a single channel of an image.

    Args:
        row (pd.Series): A row from the DataFrame containing metadata.
        channel (int): The channel to process.
        seg_mask (np.ndarray): Segmentation mask for the image.
        camera_offset (int): Offset to subtract from the image background.

    Returns:
        Tuple[float, float, float]: Total intensity, intensity in mask, intensity outside mask.
    """
    crop = get_raw_intensity_crop(row, resolution_level=0, channel=channel)
    background_subtracted = background_subtract(crop, camera_offset=camera_offset)
    sum_proj = sum_projection(background_subtracted)

    total = total_intensity(sum_proj)
    in_mask = sum_in_mask(sum_proj, seg_mask)
    out_mask = sum_not_in_mask(sum_proj, seg_mask)

    return total, in_mask, out_mask


# Define the row processing function
def process_row(
    row: pd.Series,
    marker: str,
    nuclear_seg_channel: int,
    antibody_channel: int,
    dapi_channel: int,
    resolution_level: int = 0,
    camera_offset: int = 100,
) -> dict:
    # Nuclear segmentation mask
    seg_mask = get_segmentation_mask_crop(
        row, resolution_level=resolution_level, channel=nuclear_seg_channel
    )

    # Antibody channel
    crop_total, crop_nuc, crop_cyto = process_channel(
        row, channel=antibody_channel, seg_mask=seg_mask, camera_offset=camera_offset
    )

    # DAPI control channel
    dapi_total, dapi_nuc, dapi_cyto = process_channel(
        row, channel=dapi_channel, seg_mask=seg_mask, camera_offset=camera_offset
    )

    # Normalized values
    epsilon = 1e-6  # add small factor so we don't divide by zero
    norm_crop = crop_total / (dapi_total + epsilon)
    norm_nuc = crop_nuc / (dapi_nuc + epsilon)
    norm_cyto = crop_cyto / (dapi_cyto + epsilon)
    norm_ratio = norm_nuc / (norm_cyto + epsilon)

    return {
        f"norm_crop_intensity_{marker}": norm_crop,
        f"norm_nuc_intensity_{marker}": norm_nuc,
        f"norm_cyto_intensity_{marker}": norm_cyto,
        f"norm_nuc_to_cyto_ratio_{marker}": norm_ratio,
    }


def add_if_cols_to_df(
    df: pd.DataFrame,
    marker: str,
    nuclear_seg_channel: int,
    antibody_channel: int,
    dapi_channel: int,
) -> pd.DataFrame:
    """
    Adds immunofluorescence columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to which the columns will be added.
        marker (str): The marker name.

    Returns:
        pd.DataFrame: The updated DataFrame with new columns.
    """
    # Process each row in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_row)(
            row, marker, nuclear_seg_channel, antibody_channel, dapi_channel
        )
        for _, row in df.iterrows()
    )

    # Combine results with the original DataFrame
    results_df = pd.DataFrame(results)
    df_new_cols = pd.concat([df, results_df], axis=1)

    return df_new_cols
