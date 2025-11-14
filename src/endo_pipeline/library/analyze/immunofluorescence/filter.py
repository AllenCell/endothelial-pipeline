import pandas as pd

from endo_pipeline.settings.image_data import (
    HOTSPOT_THRESHOLD,
    IMG_SHAPE_RESOLUTION_0_3i_X,
    IMG_SHAPE_RESOLUTION_0_3i_Y,
)


def filter_small_objects(df: pd.DataFrame, size_threshold: int = 450) -> pd.DataFrame:
    """
    Filter out nuclear segmentation objects that are smaller than a realistic size threshold.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe with track level features
    size_threshold: Float
        The minimum size threshold to keep an object

    Returns
    -------
    df_filtered: Dataframe
        The filtered dataframe with small objects removed
    """
    df_filtered = df[df["area"] >= size_threshold].copy()
    return df_filtered


def filter_edge_objects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out nuclear segmentation objects that are touching the edge of the image.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe with track level features

    Returns
    -------
    df_filtered: Dataframe
        The filtered dataframe with objects touching the edge removed
    """

    df_filtered = df[~df["touches_border"]].copy()
    return df_filtered


def filter_img_center(
    df: pd.DataFrame,
    pixels_from_edge: int = HOTSPOT_THRESHOLD,
    img_shape_x: int = IMG_SHAPE_RESOLUTION_0_3i_X,
    img_shape_y: int = IMG_SHAPE_RESOLUTION_0_3i_Y,
) -> pd.DataFrame:
    """
    Filter to nuclie at the center of the image based on centroid position.
    The laser is more uniform in the center of the image and there is roll off as you go to the edges.
    Immunoflourescence analysis is performed on zarrs with full resolution (level 0).

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe with track level features

    Returns
    -------
    df_filtered: Dataframe
        The filtered dataframe with objects not in the center removed
    """
    x_max = img_shape_x - pixels_from_edge
    y_max = img_shape_y - pixels_from_edge

    df_filtered = df[
        (df["centroid_x"] > pixels_from_edge) &
        (df["centroid_x"] < x_max) &
        (df["centroid_y"] > pixels_from_edge) &
        (df["centroid_y"] < y_max)
    ]

    return df_filtered
