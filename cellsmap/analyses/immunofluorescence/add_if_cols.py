# %%
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cellsmap.analyses.immunofluorescence.if_feature_extraction import (
    background_subtract,
    get_raw_intensity_crop,
    get_segmentation_mask_crop,
    sum_in_mask,
    sum_not_in_mask,
    sum_projection,
    total_intensity,
)


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


def get_feature_columns(marker: str, total: float, nuc: float, cyto: float) -> dict:
    """
    Generate a dictionary of feature columns for a given marker.

    Args:
        marker (str): The marker name.
        total (float): Total intensity value.
        nuc (float): Nuclear intensity value.
        cyto (float): Cytoplasmic intensity value.

    Returns:
        dict: A dictionary containing feature columns.
    """
    return {
        f"crop_intensity_{marker}": total,
        f"nuc_intensity_{marker}": nuc,
        f"cyto_intensity_{marker}": cyto,
        f"nuc_to_cyto_ratio_{marker}": nuc / (cyto + 1e-6),  # avoid division by zero
    }


# Define the row processing function
def process_row(
    row: pd.Series,
    marker: str,
    nuclear_seg_channel: int,
    antibody_channel: int,
    dapi_channel: int,
    resolution_level: int = 0,
    camera_offset: int = 100,
    add_dapi: bool = False,
) -> dict:
    """
    Processes a single row of data to calculate intensity metrics for a given marker.

    Args:
        row (pd.Series): A row of data from a DataFrame.
        antibody_channel (str): The channel name for the antibody marker.
        seg_mask (np.ndarray): The segmentation mask for identifying regions of interest.
        camera_offset (float): The camera offset value to adjust intensity calculations.
        marker (str): The name of the marker being processed.
        add_dapi (bool, optional): Whether to process the DAPI channel. Defaults to False.
        dapi_channel (str, optional): The channel name for DAPI, required if `add_dapi` is True.

    Returns:
        dict: A dictionary containing intensity metrics for the marker (and DAPI if applicable),
              including total, nuclear, cytoplasmic intensities, and nuclear-to-cytoplasmic ratio.
    """
    # Nuclear segmentation mask
    seg_mask = get_segmentation_mask_crop(
        row, resolution_level=resolution_level, channel=nuclear_seg_channel
    )

    # Antibody channel
    crop_total, crop_nuc, crop_cyto = process_channel(
        row, channel=antibody_channel, seg_mask=seg_mask, camera_offset=camera_offset
    )
    result = get_feature_columns(marker, crop_total, crop_nuc, crop_cyto)

    if add_dapi:
        dapi_total, dapi_nuc, dapi_cyto = process_channel(
            row, channel=dapi_channel, seg_mask=seg_mask, camera_offset=camera_offset
        )
        result.update(get_feature_columns("dapi", dapi_total, dapi_nuc, dapi_cyto))

    return result


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
        nuclear_seg_channel (int): The channel for nuclear segmentation.
        antibody_channel (int): The channel for the antibody marker.
        dapi_channel (int): The channel for DAPI.

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
