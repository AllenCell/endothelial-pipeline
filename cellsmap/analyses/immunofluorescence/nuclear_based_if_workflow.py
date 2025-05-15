# %%
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from bioio import BioImage
from skimage.measure import regionprops

from cellsmap.analyses.immunofluorescence.add_if_cols import add_if_cols_to_df
from cellsmap.analyses.immunofluorescence.if_feature_extraction import (
    background_subtract,
    get_crop_size,
    get_raw_intensity_crop,
    get_segmentation_mask_crop,
    sum_projection,
)
from cellsmap.analyses.immunofluorescence.plots import (
    plot_intensity_distribution,
    projection_image,
)
from cellsmap.util import dataset_io, manifest_io, set_output


# %%
def get_full_segmentation_path(dataset: str, position: str) -> Path:
    """Get the full path to the segmentation image for a given dataset and position."""
    p = dataset_io.extract_P(position)
    path_to_nuc_seg = Path(dataset_io.get_nuclear_prediction_path(dataset, p))
    fname = f"{dataset}_{position}_T0.ome.tiff"
    full_path = path_to_nuc_seg / fname

    if not full_path.exists():
        raise FileNotFoundError(f"Segmentation mask file not found: {full_path}")

    return full_path


def get_segmentation_image(dataset: str, position: str, frame: int) -> np.ndarray:
    """Load the segmentation image for a given dataset and position."""
    full_path = get_full_segmentation_path(dataset, position)
    seg_img = BioImage(str(full_path))
    seg_yx = seg_img.get_image_data("YX", C=0, T=0, Z=0)
    return seg_yx


def extract_region_properties(
    seg_2d: np.ndarray, position: str, frame: int, full_path: Path
) -> pd.DataFrame:
    """Extract region properties from the segmentation image."""
    props = regionprops(seg_2d)
    results = [
        (
            position,
            frame,
            full_path,
            prop.label,
            round(prop.centroid[1], 2),
            round(prop.centroid[0], 2),
            prop.area,
        )
        for prop in props
    ]
    columns = [
        "position",
        "image_index",
        "seg_path",
        "img_label",
        "centroid_x",
        "centroid_y",
        "nuclear_area",
    ]
    return pd.DataFrame(results, columns=columns)


def add_metadata_to_df(
    df: pd.DataFrame, dataset: str, seg_2d: np.ndarray, position: str
) -> pd.DataFrame:
    """Add metadata columns to the DataFrame."""
    df["dataset"] = dataset
    p = dataset_io.extract_P(position)
    fname = dataset_io.get_zarr_name(dataset, p)
    zarr_path = dataset_io.get_zarr_path(dataset, fname)
    df["zarr_path"] = zarr_path[fname]
    df["image_size_x"] = seg_2d.shape[1]
    df["image_size_y"] = seg_2d.shape[0]
    return df


def filter_centroids_near_edge(
    df: pd.DataFrame, resolution_level: Literal[0, 1]
) -> pd.DataFrame:
    """Filter out centroids near the edges of the image."""
    buffer_x = get_crop_size(resolution_level)
    buffer_y = get_crop_size(resolution_level)

    filtered_df = df[
        (df["centroid_x"] > buffer_x)
        & (df["centroid_x"] < (df["image_size_x"] - buffer_x))
        & (df["centroid_y"] > buffer_y)
        & (df["centroid_y"] < (df["image_size_y"] - buffer_y))
    ]
    return filtered_df


def add_start_coords(df: pd.DataFrame, resolution_level: Literal[0, 1]) -> pd.DataFrame:
    """Add start coordinates to the DataFrame."""
    shift_x = get_crop_size(resolution_level) // 2
    shift_y = get_crop_size(resolution_level) // 2
    df["start_x"] = (df["centroid_x"] - shift_x).astype(int)
    df["start_y"] = (df["centroid_y"] + shift_y).astype(int)
    return df


def process_positions(
    dataset: str,
    positions: list,
    marker: str,
    resolution_level: Literal[0, 1],
) -> pd.DataFrame:
    """Process all positions for a given dataset."""
    all_data = []
    for position in positions:
        seg_2d = get_segmentation_image(dataset, position, frame=0)
        full_path = get_full_segmentation_path(dataset, position)
        df = extract_region_properties(seg_2d, position, frame=0, full_path=full_path)
        df = add_metadata_to_df(df, dataset, seg_2d, position)
        df = filter_centroids_near_edge(df, resolution_level)
        df = add_start_coords(df, resolution_level)
        df["frame_number"] = df["image_index"]

        df = add_if_cols_to_df(
            df,
            marker=marker,
            nuclear_seg_channel=0,
            antibody_channel=3,
            dapi_channel=2,
            resolution_level=resolution_level,
        )

        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


# Main workflow
def main() -> None:
    dataset = "20250122_SMAD1"
    marker = "SMAD1"
    positions = ["P5", "P6", "P7", "P8", "P9"]
    resolution_level = 0

    output_dir = set_output.get_output_path("smad1_analysis")

    df_results = process_positions(dataset, positions, marker, resolution_level)
    df_results.to_csv(output_dir + f"{dataset}_nuclear_results.csv", index=False)


if __name__ == "__main__":
    main()
# %%
