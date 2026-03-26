# %%
import logging

from endo_pipeline.cli.apps import WorkflowOptions, apply_workflow_options
from endo_pipeline.configs import (
    PositionAnnotation,
    TimepointAnnotation,
    get_datasets_in_collection,
    load_dataset_collection_config,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    filter_dataframe_by_annotations,
    filter_dataframe_by_track_length,
    split_dataset_by_flow,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
    TRACK_METADATA_COLUMNS_TO_KEEP,
)
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
column_names = list(DYNAMICS_COLUMN_NAMES)
variable_labels_dict = {
    col: get_label_for_column(col).replace("polar ", "") for col in column_names
}
columns_to_compute = [*METADATA_COLUMNS_TO_KEEP, *column_names]

dataset_collection_name = "flow_switch"
dataset_collection_config = load_dataset_collection_config(dataset_collection_name)

# Load dataframe manifest for the features to be used in flow field
# estimation and analysis.
if crop_pattern == "grid":
    base_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
else:
    # NOTE: current tracked feature dataframe has no filtering applied, so needs additional
    # timepoint + position + "is_included" track filtering applied before analysis
    feature_dataframe_manifest_name = "pc_diffae_tracked_seg_features"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

valid_dataset_options = list(feature_dataframe_manifest.locations.keys())
dataset_names = get_datasets_in_collection(dataset_collection_name, subset=valid_dataset_options)


# %%
# loop over datasets in collection and do analysis
for dataset_name in dataset_names:
    dataset_config = load_dataset_config(dataset_name)
    # load dataframe and perform additional filtering (remove
    # non-steady-state timepoints based on annotations), computing
    # only the columns needed for flow field estimation and analysis to save memory.
    df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    # start with default metadata columns to keep
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP, *column_names]
    if crop_pattern == "tracked":
        # also keep track ID and track length columns for tracked crops
        columns_to_compute = [*columns_to_compute, *TRACK_METADATA_COLUMNS_TO_KEEP]
    df_ = df[columns_to_compute].compute()
    df_steady_state = filter_dataframe_by_annotations(
        df_,
        load_dataset_config(dataset_name),
        timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
    )

    if crop_pattern == "tracked":
        # additional filtering currently necessary for loading pattern with
        # tracked crops; this will be updated once the dataframe tracking
        # structure is standardized
        df_steady_state = df_steady_state[df_steady_state[ColumnName.SegDataFilters.IS_INCLUDED]]
        df_steady_state = filter_dataframe_by_annotations(
            df_steady_state,
            dataset_config,
            timepoint_annotations=list(TimepointAnnotation),
            position_annotations=list(PositionAnnotation),
        )
        # also filter out tracks that are too short for reliable flow field estimation and analysis
        df_steady_state = filter_dataframe_by_track_length(
            df_steady_state, ColumnName.TRACK_LENGTH, minimum_track_length=100
        )
    # split dataset by flow condition (e.g., 6 and 20 dyn/cm2 for the 6 to 20 dyn/cm2 dataset)
    dfs_by_flow, shear_stress_list = split_dataset_by_flow(df, dataset_config)
    for df_flow, shear_stress in zip(dfs_by_flow, shear_stress_list, strict=True):
        logger.info(f"Processing dataset {dataset_name} with shear stress {shear_stress}")
        logger.info(f"Number of samples: {len(df_flow)}")
        # do analysis here
