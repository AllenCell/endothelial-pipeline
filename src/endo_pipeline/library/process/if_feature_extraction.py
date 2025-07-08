from typing import Any

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, shannon_entropy

from src.endo_pipeline.configs import dataset_io
from src.endo_pipeline.library.process.image_processing import (
    background_subtract,
    max_proj,
    normalize_image,
    sum_proj,
)

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


def identify_edge_nuclei(label_mask: np.ndarray, bounding_box: tuple) -> bool:
    """
    Identify if a labeled region touches the border of the image.
    Args:
        label_mask (np.ndarray): The labeled image used to identify the boundary of the FOV.
        bounding_box (tuple): A tuple representing the bounding box of the labeled region
    Returns:
        bool: True if the labeled region touches the border of the image, False otherwise.
    """
    image_height, image_width = label_mask.shape
    minr, minc, maxr, maxc = bounding_box

    touches_border = minr == 0 or minc == 0 or maxr == image_height or maxc == image_width

    return touches_border


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
            "perimeter": p.perimeter,
            "solidity": p.solidity,
            "major_axis_length": p.major_axis_length,
            "minor_axis_length": p.minor_axis_length,
            "aspect_ratio": (
                p.major_axis_length / p.minor_axis_length if p.minor_axis_length > 0 else np.nan
            ),
            "eccentricity": p.eccentricity,
            "touches_border": identify_edge_nuclei(label_image, p.bbox),
        }
        for p in props
    ]


def calculate_glcm_features(image: np.ndarray) -> dict:
    """
    Calculate GLCM ("grey-level co-occurrence matrix") features are calculated
    to describe the texture of an image.

    Args:
        image (np.ndarray): Input image, should be a 2D grayscale image.

    Returns:
        dict: A dictionary containing GLCM features
        (ie. contrast, correlation, energy, and homogeneity)
    """
    normalized_image = normalize_image(image)

    glcm = graycomatrix(
        normalized_image,
        distances=[1],  # distance of 1 pixel
        angles=[0, np.pi / 4, np.pi / 2],  # horizontal, diagonal, vertical
        symmetric=True,
        normed=True,
    )
    glcm_features = {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
    }
    return glcm_features


def compute_projection_properties(p: Any, channel: str, proj_type: str) -> dict[str, float]:
    """
    Compute statistical properties of a given projection for a specific channel.

    Args:
        p (Any): A region property object containing intensity image data.
        channel (str): The name of the channel.
        proj_type (str): The type of projection (e.g., "sum", "max").

    Returns:
        Dict[str, float]: A dictionary containing computed statistical properties for the given
                         projection. Keys are formatted as "{channel}_{statistic}_{proj_type}_proj".
    """
    glcm_features = calculate_glcm_features(p.intensity_image)

    return {
        f"{channel}_std_{proj_type}_proj": np.std(p.intensity_image),
        f"{channel}_sum_{proj_type}_proj": np.sum(p.intensity_image),
        f"{channel}_mean_{proj_type}_proj": p.mean_intensity,
        f"{channel}_median_{proj_type}_proj": np.median(p.intensity_image),
        f"{channel}_max_{proj_type}_proj": p.max_intensity,
        f"{channel}_min_{proj_type}_proj": p.min_intensity,
        f"{channel}_25th_percentile_{proj_type}_proj": np.percentile(p.intensity_image, 25),
        f"{channel}_50th_percentile_{proj_type}_proj": np.percentile(p.intensity_image, 50),
        f"{channel}_75th_percentile_{proj_type}_proj": np.percentile(p.intensity_image, 75),
        f"{channel}_entropy_{proj_type}_proj": shannon_entropy(p.intensity_image),
        f"{channel}_glcm_contrast_{proj_type}_proj": glcm_features["contrast"],
        f"{channel}_glcm_correlation_{proj_type}_proj": glcm_features["correlation"],
        f"{channel}_glcm_energy_{proj_type}_proj": glcm_features["energy"],
        f"{channel}_glcm_homogeneity_{proj_type}_proj": glcm_features["homogeneity"],
    }


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
    max_projection = max_proj(background_subtracted, axis=2)[0, 0, :, :]  # return 2D image YX

    # Extract region properties using the sum projection as the intensity image
    props_sum_proj = regionprops(label_image, intensity_image=sum_projection)
    props_max_proj = regionprops(label_image, intensity_image=max_projection)

    # Return a list of dictionaries containing the extracted properties for both projections
    return [
        {
            "label": p_sum.label,
            **compute_projection_properties(p_sum, channel, "sum"),
            **compute_projection_properties(p_max, channel, "max"),
        }
        for p_sum, p_max in zip(props_sum_proj, props_max_proj, strict=True)
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
