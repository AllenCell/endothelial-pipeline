import pandas as pd


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
