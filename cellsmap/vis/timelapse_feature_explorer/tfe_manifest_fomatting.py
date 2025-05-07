import ast
from pathlib import Path

import pandas as pd

from cellsmap.vis.timelapse_feature_explorer.backdrop_images import (
    add_backdrop_fname_to_manifest,
    generate_backdrops,
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
    # Ensure 'centroid' column is properly parsed
    # df["centroid"] = df["centroid"].apply(
    #     lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    # )
    # df["centroid_x"] = df["centroid"].apply(lambda x: x[1] if x else None)
    # df["centroid_y"] = df["centroid"].apply(lambda x: x[0] if x else None)

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
        ],
        inplace=True,
    )

    return df
