import ast
from pathlib import Path

import numpy as np
import pandas as pd
from colorizer_data import FeatureInfo

from cellsmap.analyses.track_data_plots import (
    add_filter_columns,
    calculate_derived_data_dynamics_dependent,
)
from cellsmap.util.manifest_preprocessing.diffae_feature_preprocessing import (
    project_manifest_to_pcs,
)
from cellsmap.util.manifest_preprocessing.manifest_pca import fit_pca
from cellsmap.vis.timelapse_feature_explorer.backdrop_images import (
    add_backdrop_fname_to_manifest,
)
from cellsmap.vis.timelapse_feature_explorer.feature_info import LABEL_MAP


def update_manifest_for_tfe(
    df: pd.DataFrame, dataset: str, position: int, output_dir: Path
) -> pd.DataFrame:
    """
    Update the manifest DataFrame for TFE by adding necessary columns.

    Args:
        df (pd.DataFrame): The input manifest DataFrame.
        dataset (str): The dataset name.
        position (int): The position identifier.
        output_dir (Path): The output directory for backdrops.

    Returns:
        pd.DataFrame: The updated manifest DataFrame.
    """
    # Add dataset and position columns
    df["dataset"] = dataset
    df["position"] = position

    # Generate segmentation image filenames
    df["seg_image"] = (
        df["dataset"]
        + "_P"
        + df["position"].astype(str)
        + "_T"
        + df["T"].astype(str)
        + ".ome.tiff"
    )

    # Add backdrop filenames to the manifest
    df = add_backdrop_fname_to_manifest(
        df,
        dataset,
        position,
        ["bf_slice", "bf_std_dev", "gfp_max_proj"],
        output_dir=output_dir / "backdrops",
    )

    # Add track ID and drop unnecessary columns
    df["tid"] = df["track_id"]
    df.drop(
        columns=[
            "centroid",
            "T",
            "reference_index",
            "matched_query_label",
            "optimized_metric_value",
            "matching_method",
            "source_tracking_table_path",
            "filepath_raw_image",
            "scene_index",
            "neighboring_cell_labels",
            "edge_labels",
            "node_labels",
            "node_pair_labels",
            "zarr_path",
            "image_size_x",
            "image_size_y",
            "EGFP_channel_index_zarr",
            "brightfield_channel_index_zarr",
            "num_tracks_at_T",
            "centroid_x",
            "centroid_y",
            "cell_centroid",
            "cell_label",
            "dalignment_dt_deg_rel_to_flow",
        ],
        inplace=True,
    )

    return df


def add_track_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add track duration to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The updated DataFrame with track duration in frames.
    """
    df["track_duration"] = df.groupby("track_id")["image_index"].transform(
        "max"
    ) - df.groupby("track_id")["image_index"].transform("min")
    return df


def add_intensity_mean_pcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform PCA on the intensity mean features and add the projected features to the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The updated DataFrame with PCA features.
    """

    feat_cols = [f"intensity_mean_feat_{i}" for i in range(8)]

    nan_rows = df[df[feat_cols].isna().any(axis=1)]  # drop nans in order to run the pca
    df_cleaned = df.dropna(subset=feat_cols)

    pca = fit_pca()
    df_projected = project_manifest_to_pcs(
        df_cleaned, pca, overwrite_feature_columns=False, feat_cols=feat_cols
    )

    df_result = pd.concat([df_projected, nan_rows], ignore_index=True)

    assert (
        df.shape[0] == df_result.shape[0]
    ), "Shape mismatch dropping and merging back NaN rows"

    return df_result


def add_dynamic_features_with_filtering(df: pd.DataFrame) -> pd.DataFrame:
    df = add_filter_columns(df, out_dir=None)
    df_filtered_rows = df[df["filter_global"] == True]
    df_keep = df[df["filter_global"] == False]
    df_calc = calculate_derived_data_dynamics_dependent(df_keep)

    df_result = pd.concat([df_calc, df_filtered_rows], ignore_index=True)

    assert (
        df.shape[0] == df_result.shape[0]
    ), "Shape mismatch dropping and merging back filtered rows"
    return df_result


def add_feauture_metadata(df: pd.DataFrame) -> dict:
    feature_info = {}

    # Iterate through the label_map to populate feature_info
    for feature, label in LABEL_MAP.items():
        feature_info[feature] = FeatureInfo(
            label=label,
        )

    # Return the feature_info dictionary
    return feature_info
