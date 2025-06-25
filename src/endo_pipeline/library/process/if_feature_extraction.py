from typing import Any

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from cellsmap.util import dataset_io
from src.endo_pipeline.library.process.image_processing import background_subtract, sum_proj

IF_CHANNELS = ["NucViolet", "SOX17", "SMAD1", "NR2F2"]
NUC_SEG_TYPE = "nuclear_stain_seg_path"


def get_labeled_nuclei(
    dataset: str, position: int, timepoint: int, nuc_seg_type: str
) -> np.ndarray:
    """
    Generate a labeled nuclei image for a given dataset, position, and timepoint.

    Args:
        dataset (str): The name of the dataset.
        position (int): The position index within the dataset.
        timepoint (int): The timepoint index for the dataset.
        nuc_seg_type (str): The type of nuclear segmentation to use.

    Returns:
        np.ndarray: A labeled image where each connected component is assigned a unique integer label.
    """
    seg_image = dataset_io.load_nuclei_prediction(
        dataset_name=dataset,
        position=position,
        T=timepoint,
        nuc_seg_type=nuc_seg_type,
        dim_order="YX",
    )
    return label(seg_image)


def extract_morphological_props(
    label_image: np.ndarray, dataset: str, position: int
) -> list[dict[str, Any]]:
    """
    Extract morphological properties from a labeled image.

    Args:
        label_image (np.ndarray): A labeled image where each connected component is assigned a
            unique integer label.
        dataset (str): The name of the dataset.
        position (int): The position index within the dataset.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing morphological properties for
            a labeled region.
    """
    props = regionprops(label_image)
    return [
        {
            "dataset": dataset,
            "position": position,
            "label": p.label,
            "area": p.area,
            "centroid_y": p.centroid[0],
            "centroid_x": p.centroid[1],
            "eccentricity": p.eccentricity,
        }
        for p in props
    ]


def extract_if_channel_props(
    label_image: np.ndarray, channel: str, raw_image: np.ndarray
) -> list[dict[str, float]]:
    """
    Subtract background and compute intensity-based properties for a specific immunofluorescence
    channel.

    Args:
        label_image (np.ndarray): A labeled image where each connected component is assigned a
            unique integer label. 2D image YX.
        channel (str): The name of the channel.
        raw_image (np.ndarray): The raw 3D image data for the given channel in TZYX format.

    Returns:
        List[Dict[str, float]]: A list of dictionaries, each containing intensity-based properties
            for a labeled region.
    """
    # Perform background subtraction on the raw image
    background_subtracted = background_subtract(raw_image)

    # Compute the sum projection along the Z-axis
    sum_projection = sum_proj(background_subtracted, axis=2)[0, 0, :, :]  # return 2D image YX

    # Extract region properties using the sum projection as the intensity image
    props = regionprops(label_image, intensity_image=sum_projection)

    # Return a list of dictionaries containing the extracted properties
    return [
        {
            "label": p.label,
            f"{channel}_sum_proj_std": np.std(p.intensity_image),
            f"{channel}_total_sum_proj": np.sum(p.intensity_image),
            f"{channel}_mean_sum_proj": p.mean_intensity,
            f"{channel}_max_sum_proj": p.max_intensity,
            f"{channel}_min_sum_proj": p.min_intensity,
        }
        for p in props
    ]


def process_position(
    dataset: str,
    position: int,
    if_channels: list[str],
    timepoint: int,
    nuc_seg_type: str,
) -> pd.DataFrame:
    """
    Process a specific position in the dataset to extract morphological and
    intensity-based properties for specified channels.

    Args:
        dataset (str): The name of the dataset.
        position (int): The position index within the dataset.
        if_channels (List[str]): A list of channel names to extract intensity-based properties.
        timepoint (int): The timepoint index for the dataset.
        nuc_seg_type (str): The type of nuclear segmentation to use.

    Returns:
        pd.DataFrame: A DataFrame containing the combined morphological and intensity-based
            properties.
    """
    label_image = get_labeled_nuclei(dataset, position, timepoint, nuc_seg_type)
    morph_props = extract_morphological_props(label_image, dataset, position)
    df_position = pd.DataFrame(morph_props)

    for channel in dataset_io.get_channel_names(dataset):
        if channel in if_channels:
            raw_image = dataset_io.load_dataset_position_as_dask_array(
                dataset, position, channels=[channel]
            )
            channel_props = extract_if_channel_props(label_image, channel, raw_image)
            df_channel: pd.DataFrame = pd.DataFrame(channel_props)

            df_position = pd.merge(
                df_position, df_channel, on="label", how="left", validate="one_to_one"
            )

    return df_position


def run_nuclei_feature_extraction(
    dataset: str,
    if_channels: list[str] = IF_CHANNELS,
    timepoint: int = 0,
    nuc_seg_type: str = NUC_SEG_TYPE,
) -> pd.DataFrame:
    """
    Run nuclei feature extraction for all positions in the dataset.

    Args:
        dataset (str): The name of the dataset.
        if_channels (List[str]): A list of channel names to extract intensity-based properties.
            Defaults to the current list of immunofluorescence channels.
        timepoint (int): The timepoint index for the dataset.
            Default is 0, as IF data is only at timepoint 0.
        nuc_seg_type (str): The type of nuclear segmentation to use.
            Default is "nuclear_stain_seg_path", IF data uses the segmentations
            generated from the nuclear stain.

    Returns:
        pd.DataFrame: A single DataFrame containing the extracted features for all positions.
    """
    n_positions = dataset_io.get_total_number_of_positions(dataset)
    df_all_positions = [
        process_position(dataset, pos, if_channels, timepoint, nuc_seg_type)
        for pos in range(n_positions)
    ]
    # Concatenate all DataFrames into a single DataFrame
    df_dataset = pd.concat(df_all_positions, ignore_index=True)
    return df_dataset
