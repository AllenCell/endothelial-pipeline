from pathlib import Path

import numpy as np
import pandas as pd
from bioio_base.types import PhysicalPixelSizes
from tqdm import tqdm

from endo_pipeline.library.process.general_image_preprocessing import save_image_output
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.image_data import (
    IMG_SHAPE_RESOLUTION_1_3i_X,
    IMG_SHAPE_RESOLUTION_1_3i_Y,
    PIXEL_SIZE_3i_20x,
)


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
    """
    Create and save grid segmentation images to the specified output directory
    for each position and timepoint in the grid-based DiffAE dataframe.

    When creating the segmentation image, assign the crop_index from grid_df to
    be the "segmentation" label. We will use the crop index as the segmentation
    ID. The segmentation labels in the image will be equal to 1 + the crop_index
    from the grid_df. This is because 0 must be reserved for the background.
    """

    crop_index_slices = make_crop_index_to_slice_mapping(grid_df)

    # check that the crops will fit in an initialized image
    img_shape = (img_shape_y, img_shape_x)
    grid_seg = np.zeros(img_shape, dtype=np.uint16)

    if (
        grid_df[Column.DiffAEData.END_X].max() > img_shape_x
        or grid_df[Column.DiffAEData.END_Y].max() > img_shape_y
    ):
        raise ValueError(f"Grid crop locations exceed expected image shape of '{img_shape}'")

    # we can probably do the multiprocessing at the position level
    for position, df in grid_df.groupby(Column.POSITION):
        out_subdir = out_dir / f"P{position}"
        out_subdir.mkdir(exist_ok=True)

        # each position has a unique set of crop index labels
        # intialize an empty image to hold the segmentation labels
        grid_seg = np.zeros(img_shape, dtype=np.uint16)

        for crop_i in df[Column.CROP_INDEX].unique():
            grid_seg[crop_index_slices[crop_i]] = (
                crop_i + 1
            )  # add 1 so that background is 0 and first crop is label 1

        # save the grid segmentation image for this position and timepoint
        for timepoint in tqdm(
            df[Column.TIMEPOINT].unique(),
            desc=f"Saving grid segmentation for position {position}",
        ):
            fname = f"P{position}_T{timepoint}_grid_segmentation.ome.tiff"

            resolution_level = np.unique(df[Column.DiffAEData.RESOLUTION]).item()
            px_res_xy = pixel_size * 2**resolution_level
            px_res = PhysicalPixelSizes(Z=None, Y=px_res_xy, X=px_res_xy)

            metadata = {
                "image_name": f"grid_segmentation_{timepoint}",
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
