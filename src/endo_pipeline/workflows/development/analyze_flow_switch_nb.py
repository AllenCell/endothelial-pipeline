# %%
import logging

from endo_pipeline.cli.apps import WorkflowOptions, apply_workflow_options
from endo_pipeline.configs import (
    get_datasets_in_collection,
    load_dataset_collection_config,
    load_dataset_config,
)
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    split_dataset_by_flow,
)
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.dynamics_workflows import NUM_PCS_TO_FIT_FOR_DYNAMICS, RESCALE_THETA
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)
apply_workflow_options(WorkflowOptions(verbose=True))
# %%
model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
run_name = DEFAULT_MODEL_RUN_NAME
crop_pattern = "grid"

dataset_collection_name = "flow_switch"
dataset_collection_config = load_dataset_collection_config(dataset_collection_name)

# get dataframe manifest for grid-based crop features
model_manifest = load_model_manifest(model_manifest_name)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, run_name, crop_pattern=crop_pattern
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

valid_dataset_options = list(dataframe_manifest.locations.keys())
dataset_names = get_datasets_in_collection(dataset_collection_name, subset=valid_dataset_options)

# fit PCA - ALWAYS on grid-based crop features
dataframe_manifest_name_for_pca = get_feature_dataframe_manifest_name(
    model_manifest, run_name, crop_pattern="grid"
)
pca = fit_pca(
    dataframe_manifest_name=dataframe_manifest_name_for_pca, num_pcs=NUM_PCS_TO_FIT_FOR_DYNAMICS
)


# %%
# loop over datasets in collection and do analysis
for dataset_name in dataset_names:
    dataset_config = load_dataset_config(dataset_name)
    df = get_dataframe_for_dynamics_workflows(
        dataset_name,
        dataframe_manifest,
        pca=pca,
        include_cell_piling=False,
        include_not_steady_state=False,
        crop_pattern=crop_pattern,
        compute_polar=True,
        rescale_theta=RESCALE_THETA,
    )
    # split dataset by flow condition (e.g., 6 and 20 dyn/cm2 for the 6 to 20 dyn/cm2 dataset)
    dfs_by_flow, shear_stress_list = split_dataset_by_flow(df, dataset_config)
    for df_flow, shear_stress in zip(dfs_by_flow, shear_stress_list, strict=True):
        logger.info(f"Processing dataset {dataset_name} with shear stress {shear_stress}")
        logger.info(f"Number of samples: {len(df_flow)}")
        # do analysis here
