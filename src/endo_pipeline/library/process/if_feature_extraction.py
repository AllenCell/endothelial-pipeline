from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, shannon_entropy

from endo_pipeline.configs import DatasetConfig, dataset_io
from endo_pipeline.io import load_image
from endo_pipeline.library.process.image_processing import (
    background_subtract,
    max_proj,
    normalize_image,
    sum_proj,
)
from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

IF_CHANNELS = ["NucViolet", "SOX17", "SMAD1", "NR2F2"]
NUC_SEG_TYPE = "nuclear_stain_seg"


def get_labeled_nuclei(
    dataset_config: DatasetConfig, position: int, nuc_seg_type: str
) -> np.ndarray:
    """
    Generate a labeled nuclei image for a given dataset and position.

    Parameters
    ----------
    dataset_config : DatasetConfig
        The configuration object for the dataset.
    position : int
        The position index within the dataset.
    nuc_seg_type : str
        The type of nuclear segmentation to use.

    Returns
    -------
    np.ndarray
        A labeled image where each connected component is assigned a unique integer label.
    """
    seg_manifest = load_image_manifest(nuc_seg_type)
    seg_location = get_image_location_for_dataset(seg_manifest, dataset_config, position)
    seg_image = load_image(seg_location, squeeze=True, compute=True)

    return label(seg_image)


def identify_edge_nuclei(label_mask: np.ndarray, bounding_box: tuple) -> bool:
    """
    Determine if a labeled region touches the border of the image.

    Parameters
    ----------
    label_mask : np.ndarray
        The labeled image used to identify the boundary of the field of view (FOV).
    bounding_box : tuple
        A tuple representing the bounding box of the labeled region.

    Returns
    -------
    bool
        True if the labeled region touches the border of the image, False otherwise.
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

    Parameters
    ----------
    label_image : np.ndarray
        A labeled image where each connected component is assigned a unique integer label.
    dataset : str
        The name of the dataset.
    position : int
        The position index within the dataset.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing morphological properties for a labeled region.
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
    Calculate GLCM (grey-level co-occurrence matrix) features to describe the texture of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image, should be a 2D grayscale image.

    Returns
    -------
    dict
        A dictionary containing GLCM features, including contrast, correlation, energy, and homogeneity.
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

    Parameters
    ----------
    p : Any
        A region property object containing intensity image data.
    channel : str
        The name of the channel.
    proj_type : str
        The type of projection (e.g., "sum", "max").

    Returns
    -------
    dict of str to float
        A dictionary containing computed statistical properties for the given projection.
        Keys are formatted as "{channel}_{statistic}_{proj_type}_proj".
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
    label_image: np.ndarray, channel: str, raw_image: da.Array
) -> list[dict[str, float]]:
    """
    Subtract background and compute intensity-based properties for a specific immunofluorescence channel.

    Parameters
    ----------
    label_image : np.ndarray
        A labeled image where each connected component is assigned a unique integer label.
        The image is 2D in YX format.
    channel : str
        The name of the channel.
    raw_image : np.ndarray
        The raw 3D image data for the given channel in TZYX format.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing intensity-based properties for a labeled region.
    """
    # Perform background subtraction on the raw image
    background_subtracted = background_subtract(raw_image)

    # Compute the sum projection along the Z-axis
    sum_projection = sum_proj(background_subtracted, axis=0)
    max_projection = max_proj(background_subtracted, axis=0)

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
    dataset_config: DatasetConfig,
    position: int,
    if_channels: list[str],
    nuc_seg_type: str,
) -> pd.DataFrame:
    """
    Process a specific position in the dataset to extract morphological and
    intensity-based properties for specified channels.

    Parameters
    ----------
    dataset_config : DatasetConfig
        The configuration object for the dataset.
    position : int
        The position index within the dataset.
    if_channels : list of str
        A list of channel names to extract intensity-based properties.
    nuc_seg_type : str
        The type of nuclear segmentation to use.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the combined morphological and intensity-based properties.
    """
    label_image = get_labeled_nuclei(dataset_config, position, nuc_seg_type)
    morph_props = extract_morphological_props(label_image, dataset_config.name, position)
    df_position = pd.DataFrame(morph_props)

    for channel in dataset_io.get_channel_names(dataset_config.name):
        if channel in if_channels:
            zarr_manifest = load_image_manifest("image_zarr")
            zarr_location = get_image_location_for_dataset(zarr_manifest, dataset_config, position)
            raw_image = load_image(zarr_location, channels=[channel], squeeze=True)
            channel_props = extract_if_channel_props(label_image, channel, raw_image)
            df_channel: pd.DataFrame = pd.DataFrame(channel_props)

            df_position = pd.merge(
                df_position, df_channel, on="label", how="left", validate="one_to_one"
            )

    return df_position


def run_nuclei_feature_extraction(
    dataset_config: DatasetConfig,
    positions: list[int],
    if_channels: list[str] = IF_CHANNELS,
    nuc_seg_type: str = NUC_SEG_TYPE,
) -> pd.DataFrame:
    """
    Run nuclei feature extraction for all positions in the dataset.

    Parameters
    ----------
    dataset_config : DatasetConfig
        The config for the dataset.
    positions : list of int
        A list of positions in the dataset to process.
    if_channels : list of str, optional
        A list of channel names to extract intensity-based properties.
        Defaults to the current list of immunofluorescence channels.
    nuc_seg_type : str, optional
        The type of nuclear segmentation to use. Defaults to "nuclear_stain_seg".
        IF data uses the segmentations generated from the nuclear stain.

    Returns
    -------
    pd.DataFrame
        A single DataFrame containing the extracted features for all positions.
    """

    df_all_positions = [
        process_position(dataset_config, pos, if_channels, nuc_seg_type) for pos in positions
    ]
    # Concatenate all DataFrames into a single DataFrame
    df_dataset = pd.concat(df_all_positions, ignore_index=True)
    return df_dataset
