from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE, ColumnName
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
run_name = DEFAULT_MODEL_RUN_NAME
dataset_name = "20250618_20X"


dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, run_name, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=MAX_PCS_TO_COMPUTE)

grid_df = get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest, pca)
feat_cols = [col for col in grid_df.columns if ColumnName.LATENT_FEATURE_PREFIX in col]
grid_df = grid_df.drop(columns=feat_cols)

# when creating the segmentation image assign the crop_index from grid_df
# to be the "segmentation" label. We will use the crop index as the
# segmentation ID.
