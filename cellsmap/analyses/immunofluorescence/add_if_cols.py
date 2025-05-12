# %%
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cellsmap.analyses.immunofluorescence.if_feature_extraction import (
    background_subtract,
    get_raw_intensity_crop,
    get_segmentation_mask_crop,
    mean_intensity_in_mask,
    mean_intensity_not_in_mask,
    median_intensity_in_mask,
    median_intensity_not_in_mask,
    sum_projection,
    total_intensity,
)


def process_channel(
    row: pd.Series, channel: int, seg_mask: np.ndarray, camera_offset: int = 100
) -> Tuple:
    """
    Processes a single channel of an image.

    Returns:
        Tuple: Total intensity, mean/median in mask, mean/median out of mask.
    """
    crop = get_raw_intensity_crop(row, resolution_level=0, channel=channel)
    background_subtracted = background_subtract(crop, camera_offset=camera_offset)
    sum_proj = sum_projection(background_subtracted)

    total = total_intensity(sum_proj)
    mean_int_in_mask = mean_intensity_in_mask(sum_proj, seg_mask)
    mean_int_not_in_mask = mean_intensity_not_in_mask(sum_proj, seg_mask)
    median_int_in_mask = median_intensity_in_mask(sum_proj, seg_mask)
    median_int_not_in_mask = median_intensity_not_in_mask(sum_proj, seg_mask)

    return (
        total,
        mean_int_in_mask,
        mean_int_not_in_mask,
        median_int_in_mask,
        median_int_not_in_mask,
    )


def get_feature_columns(
    marker: str,
    total: float,
    mean_nuc: float,
    mean_cyto: float,
    median_nuc: float,
    median_cyto: float,
) -> dict:
    """
    Generate a dictionary of feature columns for a given marker.
    """
    return {
        f"crop_intensity_{marker}": total,
        f"nuc_mean_intensity_{marker}": mean_nuc,
        f"cyto_mean_intensity_{marker}": mean_cyto,
        f"nuc_median_intensity_{marker}": median_nuc,
        f"cyto_median_intensity_{marker}": median_cyto,
        f"nuc_to_cyto_mean_ratio_{marker}": mean_nuc
        / (mean_cyto + 1e-6),  # avoid zero division
        f"nuc_to_cyto_median_ratio_{marker}": median_nuc
        / (median_cyto + 1e-6),  # avoid zero division
    }


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
    """
    seg_mask = get_segmentation_mask_crop(
        row, resolution_level=resolution_level, channel=nuclear_seg_channel
    )

    # Antibody channel processing
    crop_total, mean_nuc, mean_cyto, median_nuc, median_cyto = process_channel(
        row, channel=antibody_channel, seg_mask=seg_mask, camera_offset=camera_offset
    )
    result = get_feature_columns(
        marker, crop_total, mean_nuc, mean_cyto, median_nuc, median_cyto
    )

    if add_dapi:
        dapi_total, dapi_mean_nuc, dapi_mean_cyto, dapi_median_nuc, dapi_median_cyto = (
            process_channel(
                row,
                channel=dapi_channel,
                seg_mask=seg_mask,
                camera_offset=camera_offset,
            )
        )
        result.update(
            get_feature_columns(
                "dapi",
                dapi_total,
                dapi_mean_nuc,
                dapi_mean_cyto,
                dapi_median_nuc,
                dapi_median_cyto,
            )
        )

    return result


def add_if_cols_to_df(
    df: pd.DataFrame,
    marker: str,
    nuclear_seg_channel: int,
    antibody_channel: int,
    dapi_channel: int,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Adds immunofluorescence columns to the DataFrame using parallel processing.

    Args:
        df (pd.DataFrame): Input DataFrame.
        marker (str): Marker name for the antibody.
        nuclear_seg_channel (int): Channel index used for segmentation.
        antibody_channel (int): Channel index of the antibody.
        dapi_channel (int): Channel index for DAPI (if needed).
        n_jobs (int): Number of parallel workers. Default: -1 (all cores).

    Returns:
        pd.DataFrame: DataFrame with additional IF metrics columns.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(
            row, marker, nuclear_seg_channel, antibody_channel, dapi_channel
        )
        for _, row in df.iterrows()
    )

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
