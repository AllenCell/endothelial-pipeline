from pathlib import Path

import numpy as np
import pandas as pd
from bioio_base.types import PhysicalPixelSizes
from skimage.measure import regionprops
from tqdm import tqdm

from endo_pipeline.io import load_dataframe, load_image_from_path
from endo_pipeline.library.process.general_image_preprocessing import save_image_output
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_FEATURE_COLUMN_NAMES
from endo_pipeline.settings.image_data import (
    IMG_SHAPE_RESOLUTION_1_3i_X,
    IMG_SHAPE_RESOLUTION_1_3i_Y,
    PIXEL_SIZE_3i_20x,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def load_grid_diffae_df_for_tfe(
    dataset_name: str,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    model_run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> pd.DataFrame:
    """Load the grid-based DiffAE features for a given dataset, model manifest, and model run."""
    # load the grid-based diffae features dataframe to get the crop locations
    # and crop labels for a given dataset
    dataframe_manifest_name = f"{model_manifest_name}_{model_run_name}_grid"
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataframe_location = get_dataframe_location_for_dataset(dataframe_manifest, dataset_name)
    grid_df_ = load_dataframe(dataframe_location, delay=True)

    # don't need the feature columns for this workflow, just the crop locations
    # and labels, so we can drop them to save memory
    columns_to_compute = [col for col in grid_df_.columns if col not in DIFFAE_FEATURE_COLUMN_NAMES]
    grid_df = grid_df_[columns_to_compute].compute()

    return grid_df


def make_grid_seg_filename(position: int, timepoint: int) -> str:
    return f"{position}_T{timepoint}_grid_segmentation.ome.tiff"


def make_crop_index_to_slice_mapping(grid_df: pd.DataFrame) -> dict[int, tuple[slice, slice]]:
    """Returns a dictionary with the following structure from a grid-based DiffAE dataframe:
    {crop_index: (slice(start_y, end_y), slice(start_x, end_x))}

    This will be used to assign crop_index labels to the correct locations in the
    grid segmentation images.
    """
    crop_index_slices = dict(
        zip(
            grid_df[Column.CROP_INDEX].values,
            zip(
                map(
                    slice,
                    grid_df[Column.DiffAEData.START_Y].values,
                    grid_df[Column.DiffAEData.END_Y].values,
                ),
                map(
                    slice,
                    grid_df[Column.DiffAEData.START_X].values,
                    grid_df[Column.DiffAEData.END_X].values,
                ),
                strict=True,
            ),
            strict=True,
        )
    )
    return crop_index_slices


def create_grid_segmentation_images(
    grid_df: pd.DataFrame,
    out_dir: Path,
    img_shape_y: int = IMG_SHAPE_RESOLUTION_1_3i_Y,
    img_shape_x: int = IMG_SHAPE_RESOLUTION_1_3i_X,
    pixel_size: float = PIXEL_SIZE_3i_20x,
) -> None:
    """Creates and saves grid segmentation images to the specified output directory
    for each position and timepoint in the grid-based DiffAE dataframe.

    Note:
    The segmentation labels in the image will be equal to 1 + the crop_index from the grid_df.
    This is because 0 must be reserved for the background.
    """
    # when creating the segmentation image assign the crop_index from grid_df
    # to be the "segmentation" label. We will use the crop index as the
    # segmentation ID.
    dataset_name = np.unique(grid_df[Column.DATASET]).item()

    crop_index_slices = make_crop_index_to_slice_mapping(grid_df)

    # check that the crops will fit in an initialized image
    grid_seg = np.zeros((img_shape_y, img_shape_x), dtype=np.uint16)

    if (
        grid_df[Column.DiffAEData.END_X].max() > img_shape_x
        or grid_df[Column.DiffAEData.END_Y].max() > img_shape_y
    ):
        raise ValueError(
            f"Grid crop locations exceed expected image shape of\
            {(img_shape_y, img_shape_x)}"
        )

    # we can probably do the multiprocessing at the position level
    for pos, df in grid_df.groupby(Column.POSITION):
        out_subdir = out_dir / str(pos)
        out_subdir.mkdir(exist_ok=True)

        # each position has a unique set of crop index labels
        # intialize an empty image to hold the segmentation labels
        grid_seg = np.zeros((img_shape_y, img_shape_x), dtype=np.uint16)

        for crop_i in df[Column.CROP_INDEX].unique():
            grid_seg[crop_index_slices[crop_i]] = (
                crop_i + 1
            )  # add 1 so that background is 0 and first crop is label 1

        # save the grid segmentation image for this position and timepoint
        for tp in tqdm(
            range(np.unique(df["duration"]).item()),
            desc=f"Saving grid segmentation for {dataset_name} {pos}",
        ):
            fname = make_grid_seg_filename(pos, tp)

            resolution_level = np.unique(df[Column.DiffAEData.RESOLUTION]).item()
            px_res_xy = pixel_size * 2**resolution_level
            px_res = PhysicalPixelSizes(Z=None, Y=px_res_xy, X=px_res_xy)

            metadata = {
                "image_name": f"{dataset_name}_{tp}",
                "channel_colors": [(255, 255, 255)],
                "channel_names": ["grid_segmentation"],
                "physical_pixel_sizes": px_res,
                "dim_order": "YX",
            }
            save_image_output(
                out_path=out_subdir / fname,
                images=[grid_seg],
                images_metadata=metadata,
                dtype=np.uint16,
            )


def check_crop_indices_against_existing_segmentations(df: pd.DataFrame, out_dir: Path) -> None:
    pos = np.unique(df[Column.POSITION]).item()
    tp = np.unique(df[Column.TIMEPOINT]).item()
    fp = out_dir / pos / make_grid_seg_filename(pos, tp)

    segmentation = load_image_from_path(fp)

    segprops = regionprops(label_image=segmentation.squeeze())
    for prop in segprops:
        crop_index_from_seg = prop.label - 1
        bbox_cols = [
            Column.DiffAEData.START_Y,
            Column.DiffAEData.START_X,
            Column.DiffAEData.END_Y,
            Column.DiffAEData.END_X,
        ]

        crop_loc_matched = df[df[Column.CROP_INDEX] == crop_index_from_seg][bbox_cols] == prop.bbox
        if crop_loc_matched is False:
            raise ValueError(
                f"Crop index {crop_index_from_seg} in segmentation does not\
                    match bbox in grid_df for position {pos} and timepoint {tp}"
            )
