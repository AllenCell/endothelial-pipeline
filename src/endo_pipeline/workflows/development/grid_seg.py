import numpy as np

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.image_data import (
    IMG_SHAPE_RESOLUTION_1_3i_X,
    IMG_SHAPE_RESOLUTION_1_3i_Y,
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
            strict=False,
        ),
        strict=False,
    )
)

# intialize an empty image to hold the segmentation labels
grid_seg = np.zeros((IMG_SHAPE_RESOLUTION_1_3i_Y, IMG_SHAPE_RESOLUTION_1_3i_X), dtype=np.uint16)

# check that the crops will fit in the initialized image
if (
    grid_df.end_x.max() > IMG_SHAPE_RESOLUTION_1_3i_X
    or grid_df.end_y.max() > IMG_SHAPE_RESOLUTION_1_3i_Y
):
    raise ValueError(
        f"Grid crop locations exceed expected image shape of\
        {(IMG_SHAPE_RESOLUTION_1_3i_Y, IMG_SHAPE_RESOLUTION_1_3i_X)}"
    )

for crop_i in crop_index_slices:
    grid_seg[crop_index_slices[crop_i]] = (
        crop_i + 1
    )  # add 1 so that background is 0 and first crop is label 1
