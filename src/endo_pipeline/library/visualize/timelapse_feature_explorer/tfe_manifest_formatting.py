from pathlib import Path

import pandas as pd
from colorizer_data import FeatureInfo

from endo_pipeline.library.analyze.diffae_manifest import fit_pca, project_manifest_to_pcs
from endo_pipeline.library.visualize.timelapse_feature_explorer.backdrop_images import (
    add_backdrop_fname_to_manifest,
)
from endo_pipeline.library.visualize.timelapse_feature_explorer.feature_info import LABEL_MAP
from endo_pipeline.workflows.make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)


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
        + df["image_index"].astype(str)
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

    # Add track ID as a feature
    df["tid"] = df["track_id"]

    return df


def add_intensity_mean_pcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform PCA on the intensity mean features and add the projected features
    to the DataFrame.

    Note:
        If the associated workflow make_spatial_pc_movie.py is deleted,
        this function should also be removed.

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

    assert df.shape[0] == df_result.shape[0], "Shape mismatch dropping and merging back NaN rows"

    return df_result


def add_dynamic_features_with_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamic features can only be calculated on longer tracks that are filtered.
    For TFE we need to preserve the rows that are filtered out, so we filter them
    and then calculate the features and then merge them back in.
    """
    df_filtered_rows = df[~df["is_included"]]
    df_keep = df[df["is_included"]]
    df_calc = calculate_derived_data_dynamics_dependent(df_keep)

    df_result = pd.concat([df_calc, df_filtered_rows], ignore_index=True)

    assert (
        df.shape[0] == df_result.shape[0]
    ), "Shape mismatch dropping and merging back filtered rows"
    return df_result


def add_feature_metadata(df: pd.DataFrame) -> dict:
    """
    Only the features specified in the LABEL_MAP are added to TFE.
    Metadata right now is just the label for the feature, but this can
    be built out in the future.
    """
    feature_info = {}

    # Iterate through the label_map to populate feature_info
    for feature, label in LABEL_MAP.items():
        feature_info[feature] = FeatureInfo(
            label=label,
        )

    # Return the feature_info dictionary
    return feature_info
