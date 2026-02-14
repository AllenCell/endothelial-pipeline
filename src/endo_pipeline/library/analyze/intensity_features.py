from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.manifests import (
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_image_manifest,
)
from endo_pipeline.settings.image_data import DIMENSION_ORDER


def calculate_edge_intensity_distribution_for_segmentations(
    dataset_name, position, timepoint, df_at_tp, dim_order=DIMENSION_ORDER
):

    out_dir = Path(df_at_tp["output_dir"].unique().item())
    dataset_config = load_dataset_config(dataset_name)
    seg_manifest = load_image_manifest("cdh5_classic_seg")

    image_loc = get_zarr_location_for_position(dataset_config, position)
    raw_arr = load_image(image_loc, channels=["EGFP"], timepoints=timepoint, level=0)
    raw_arr = raw_arr.max(axis=dim_order.index("Z")).squeeze().compute()
    # voxel_size = load_image(image_loc, read=False).physical_pixel_sizes

    seg_location = get_image_location_for_dataset(seg_manifest, dataset_config, position, timepoint)
    seg_arr = load_image(seg_location, squeeze=True, compute=True)
    # seg_filepath = seg_location.path.as_posix() if seg_location.path is not None else ""

    # initialize dataframe
    df_edge_intens = pd.DataFrame(columns=df_at_tp.columns)
    df_edge_intens = df_edge_intens.drop("is_included", axis=1)

    for label, seg_record in df_at_tp.groupby("label"):
        seg_bound = find_boundaries(seg_arr == label)
        seg_bound_locs = np.where(seg_bound)
        seg_centroid = (
            seg_record["centroid_Y"].values.item(),
            seg_record["centroid_X"].values.item(),
        )

        # get the angle from each pixel in seg_bound to seg_centroid and also
        # the fluorescence intensity at each of those pixels
        angles = np.arctan2(
            seg_bound_locs[0] - seg_centroid[0],
            seg_bound_locs[1] - seg_centroid[1],
        )
        intensities = raw_arr[seg_bound_locs]

        seg_record["angle"] = [angles.tolist()]
        seg_record["intensity"] = [intensities.tolist()]

        if df_edge_intens.empty:
            df_edge_intens = seg_record.copy(deep=True)
        else:
            df_edge_intens = pd.concat([df_edge_intens, seg_record], ignore_index=True)
    out_subdir = out_dir / dataset_name / f"P{position}"
    out_subdir.mkdir(exist_ok=True, parents_ok=True)
    df_edge_intens.to_parquet(
        out_subdir / f"{dataset_name}_P{position}_T{timepoint}_edge_intensities.parquet"
    )


def calculate_edge_intensity_distribution_for_segmentations_mp(args):
    (dataset_name, position, timepoint), df_at_tp = args
    dataset_name = str(dataset_name)
    position = int(position)
    timepoint = int(timepoint)
    return calculate_edge_intensity_distribution_for_segmentations(
        dataset_name, position, timepoint, df_at_tp
    )
