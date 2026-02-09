import numpy as np
from bioio_base.types import PhysicalPixelSizes

from endo_pipeline.io import get_output_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.process.general_image_preprocessing import save_image_output
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.image_data import (
    IMG_SHAPE_RESOLUTION_1_3i_X,
    IMG_SHAPE_RESOLUTION_1_3i_Y,
    PIXEL_SIZE_3i_20x,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

dataset_name = "20250618_20X"

model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
run_name = DEFAULT_MODEL_RUN_NAME

dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, run_name, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

# load the grid-based diffae features dataframe to get the crop locations and
# crop labels for a given dataset
grid_df = get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest)

# we don't need the latent feature columns for this workflow
feat_cols = [col for col in grid_df.columns if ColumnName.LATENT_FEATURE_PREFIX in col]
grid_df = grid_df.drop(columns=feat_cols)

# when creating the segmentation image assign the crop_index from grid_df
# to be the "segmentation" label. We will use the crop index as the
# segmentation ID.

crop_index_slices = dict(
    zip(
        grid_df.crop_index.values,
        zip(
            map(slice, grid_df.start_y.values, grid_df.end_y.values),
            map(slice, grid_df.start_x.values, grid_df.end_x.values),
            strict=True,
        ),
        strict=True,
    )
)

# check that the crops will fit in an initialized image
grid_seg = np.zeros((IMG_SHAPE_RESOLUTION_1_3i_Y, IMG_SHAPE_RESOLUTION_1_3i_X), dtype=np.uint16)

if (
    grid_df.end_x.max() > IMG_SHAPE_RESOLUTION_1_3i_X
    or grid_df.end_y.max() > IMG_SHAPE_RESOLUTION_1_3i_Y
):
    raise ValueError(
        f"Grid crop locations exceed expected image shape of\
        {(IMG_SHAPE_RESOLUTION_1_3i_Y, IMG_SHAPE_RESOLUTION_1_3i_X)}"
    )

# for crop_i in crop_index_slices:
#     grid_seg[crop_index_slices[crop_i]] = (
#         crop_i + 1
#     )  # add 1 so that background is 0 and first crop is label 1

out_dir = get_output_path(__file__)
out_dir.mkdir(parents=True, exist_ok=True)
for tp in sorted(grid_df.frame_number.unique()):
    grid_df.sort_values(by="frame_number")


# we can probably do the multiprocessing at the position level
for nm, df in grid_df.groupby("position"):
    # each position has a unique set of crop index labels
    # intialize an empty image to hold the segmentation labels
    grid_seg = np.zeros((IMG_SHAPE_RESOLUTION_1_3i_Y, IMG_SHAPE_RESOLUTION_1_3i_X), dtype=np.uint16)

    for crop_i in df.crop_index.unique():
        grid_seg[crop_index_slices[crop_i]] = (
            crop_i + 1
        )  # add 1 so that background is 0 and first crop is label 1

    # save the grid segmentation image for this position and timepoint
    for tp in range(df.duration.unique().item()):
        fname = f"{dataset_name}_{tp}_grid_segmentation.ome.tiff"

        resolution_level = df.resolution_level.unique().item()
        px_res_xy = PIXEL_SIZE_3i_20x * 2**resolution_level
        px_res = PhysicalPixelSizes(Z=None, Y=px_res_xy, X=px_res_xy)

        metadata = {
            "image_name": f"{dataset_name}_{tp}",
            "channel_colors": [(255, 255, 255)],
            "channel_names": "grid_segmentation",
            "physical_pixel_sizes": px_res,
            "dim_order": "YX",
        }
        save_image_output(
            out_path=out_dir / fname, images=[grid_seg], images_metadata=metadata, dtype=np.uint16
        )
    break


for nm, df in grid_df.groupby(["crop_index", "start_y", "end_y", "start_x", "end_x"]):
    break
