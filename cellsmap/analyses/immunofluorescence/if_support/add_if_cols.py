# %%
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cellsmap.analyses.immunofluorescence.if_support.if_feature_extraction import (
    background_subtract,
    get_raw_intensity_crop,
    get_segmentation_mask_crop,
    mean_intensity_in_mask,
    mean_intensity_not_in_mask,
    median_intensity_in_mask,
    median_intensity_not_in_mask,
    sum_projection,
    total_intensity,
    total_intensity_in_mask,
)
from cellsmap.vis import get_images


def process_channel(
    row: pd.Series,
    channel_name: str,
    nuc_crop_seg_mask: np.ndarray,
    resolution_level: int,
    individual_nuc_seg_mask: Optional[np.ndarray] = None,
    camera_offset: int = 100,
) -> Tuple:
    """
    Processes a single channel of an image.

    Returns:
        Tuple: Total intensity, mean/median in mask, mean/median out of mask.
    """
    crop = get_raw_intensity_crop(
        row, resolution_level=resolution_level, channel_name=channel_name
    )
    background_subtracted = background_subtract(crop, camera_offset=camera_offset)
    sum_proj = sum_projection(background_subtracted)

    total = total_intensity(sum_proj)
    mean_int_in_mask = mean_intensity_in_mask(sum_proj, nuc_crop_seg_mask)
    mean_int_not_in_mask = mean_intensity_not_in_mask(sum_proj, nuc_crop_seg_mask)
    median_int_in_mask = median_intensity_in_mask(sum_proj, nuc_crop_seg_mask)
    median_int_not_in_mask = median_intensity_not_in_mask(sum_proj, nuc_crop_seg_mask)

    # Initialize nuclear-specific metrics
    ind_nuclear_intensity = None
    mean_int_nuclear = None
    median_int_nuclear = None

    if individual_nuc_seg_mask is not None:
        ind_nuclear_intensity = total_intensity_in_mask(
            sum_proj, individual_nuc_seg_mask
        )
        mean_int_nuclear = mean_intensity_in_mask(sum_proj, individual_nuc_seg_mask)
        median_int_nuclear = median_intensity_in_mask(sum_proj, individual_nuc_seg_mask)

    return (
        total,
        mean_int_in_mask,
        mean_int_not_in_mask,
        median_int_in_mask,
        median_int_not_in_mask,
        ind_nuclear_intensity,
        mean_int_nuclear,
        median_int_nuclear,
    )


def get_feature_columns(
    channel_name: str,
    total: float,
    mean_nuc: float,
    mean_cyto: float,
    median_nuc: float,
    median_cyto: float,
    ind_nuclear_intensity: float | None,
    mean_int_nuclear: float | None,
    median_int_nuclear: float | None,
) -> dict:
    """
    Generate a dictionary of feature columns for a given marker.
    """
    return {
        f"crop_intensity_{channel_name}": total,
        f"crop_nuc_mean_intensity_{channel_name}": mean_nuc,
        f"crop_cyto_mean_intensity_{channel_name}": mean_cyto,
        f"crop_nuc_median_intensity_{channel_name}": median_nuc,
        f"crop_cyto_median_intensity_{channel_name}": median_cyto,
        f"crop_nuc_to_cyto_mean_ratio_{channel_name}": mean_nuc
        / (mean_cyto + 1e-6),  # avoid zero division
        f"crop_nuc_to_cyto_median_ratio_{channel_name}": median_nuc
        / (median_cyto + 1e-6),  # avoid zero division
        f"nuclear_intensity_{channel_name}": ind_nuclear_intensity,
        f"nuclear_mean_intensity_{channel_name}": mean_int_nuclear,
        f"nuclear_median_intensity_{channel_name}": median_int_nuclear,
    }


def process_row(
    row: pd.Series,
    channel_name: str,
    resolution_level: Literal[0, 1],
    nuclear_seg_channel: int = 0,  # nuclear stain segmenations only have one channel
    camera_offset: int = 100,
    nuclear_centroid_crops: bool = False,
) -> dict:
    """
    Processes a single row of data to calculate intensity metrics for a given marker.
    """
    seg_mask = get_segmentation_mask_crop(
        row, resolution_level=resolution_level, channel=nuclear_seg_channel
    )
    if nuclear_centroid_crops:
        individual_nuc_seg_mask = get_segmentation_mask_crop(
            row,
            resolution_level=resolution_level,
            channel=nuclear_seg_channel,
            individual_nuc_seg_mask=True,
        )
    else:
        individual_nuc_seg_mask = None

    (
        crop_total,
        mean_nuc,
        mean_cyto,
        median_nuc,
        median_cyto,
        ind_nuclear_intensity,
        mean_int_nuclear,
        median_int_nuclear,
    ) = process_channel(
        row,
        channel_name,
        nuc_crop_seg_mask=seg_mask,
        individual_nuc_seg_mask=individual_nuc_seg_mask,
        resolution_level=resolution_level,
        camera_offset=camera_offset,
    )
    result = get_feature_columns(
        channel_name,
        crop_total,
        mean_nuc,
        mean_cyto,
        median_nuc,
        median_cyto,
        ind_nuclear_intensity,
        mean_int_nuclear,
        median_int_nuclear,
    )

    return result


def get_channels_for_if_processing(dataset_name: str) -> list[str]:
    img = get_images.get_zarr_img_for_dataset(dataset_name, 0, 1)
    channel_names = img.channel_names
    channel_names = [name for name in channel_names if name not in ["EGFP", "BF"]]
    print(f"{dataset_name} channels processing: {channel_names}")
    return channel_names


def add_if_cols_to_df(
    df: pd.DataFrame,
    channel_name: str,
    resolution_level: int,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Adds immunofluorescence columns to the DataFrame using parallel processing.

    Args:
        df (pd.DataFrame): Input DataFrame.
        channel_name (str): Marker name for the antibody or stain.
        antibody_channel (int): Channel index of the antibody.
        resolution_level (int): Resolution level to use for processing.
        n_jobs (int): Number of parallel workers. Default: -1 (all cores).

    Returns:
        pd.DataFrame: DataFrame with additional IF metrics columns.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(
            row,
            channel_name,
            resolution_level,
        )
        for _, row in df.iterrows()
    )

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
