# %%
import logging

import pandas as pd

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    project_features_to_pcs,
)
from endo_pipeline.library.analyze.migration_pc.cca_analysis import (
    apply_cca_projection,
    calculate_cca_results,
    plot_cca_projection_validation,
    plot_cca_results,
    plot_feature_correlations,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)

DESCRIPTION = "Optical flow on BF for migration coherence metric; CCA ranks top contributing PCs."

OPTICAL_FLOW_FEATURE = "optical_flow_mean_unit_vector_dt1"
UPLOAD_TO_FMS = False

datasets = get_datasets_in_collection("diffae_model_training")
output_dir = get_output_path("migration_pc_cca")

# %% Load diffae features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca, df_pca = fit_pca(num_pcs=80, return_pca_input_dataframe=True)
df_pca = project_features_to_pcs(df_pca, pca)

# %% Load optical flow features
dataframe_manifest_optical_flow = load_dataframe_manifest("optical_flow_bf")
df_of_list = []
for dataset_name in datasets:
    print(f"Processing dataset: {dataset_name} for optical flow features")
    # Get PCS
    df_dataset = get_dataframe_for_dynamics_workflows(
        dataset_name, dataframe_manifest, pca=pca, filter_dataframe=True
    )

    # Get optical flow features
    optical_flow_location = get_dataframe_location_for_dataset(
        dataframe_manifest_optical_flow, dataset_name
    )
    df_optical_flow_new = load_dataframe(optical_flow_location)

    # merge the two dataframes on the dataset, position, frame_number, start_x, start_y columns
    df_of_dataset = df_dataset.merge(
        df_optical_flow_new,
        on=["dataset", "position", "frame_number", "start_x", "start_y"],
        how="inner",
        suffixes=("", "_optical_flow"),
    )

    df_of_list.append(df_of_dataset)

# %%
df_of = pd.concat(df_of_list, ignore_index=True)
df_of_list = None  # clear list to save memory

# %%
df_of_sub = df_of.dropna(subset=[OPTICAL_FLOW_FEATURE])
# Excluded timepoints result in NaN values for the timepoint before and after the dropped timepoint.

cca_results, cca_csv_path = calculate_cca_results(
    df_of_sub,
    OPTICAL_FLOW_FEATURE,
    input_features=DIFFAE_PC_COLUMN_NAMES[:80],
    output_dir=output_dir,
    scale_cca=False,
)

# %%
plot_cca_results(cca_results, output_dir)
# %%
if UPLOAD_TO_FMS:
    dataset_configs = [load_dataset_config(dataset_name) for dataset_name in datasets]
    annotations = build_fms_annotations(
        dataset=dataset_configs, additional_notes=DESCRIPTION + OPTICAL_FLOW_FEATURE
    )
    fms_id = upload_file_to_fms(cca_csv_path, annotations=annotations, file_type="csv")

    dataframe_manifest_cca = load_dataframe_manifest("cca_weights")
    dataframe_manifest_cca.locations["80_pcs"].fmsid = fms_id
    save_dataframe_manifest(dataframe_manifest_cca)

# %%
plot_cca_projection_validation(
    df_of_sub,
    cca_results,
    output_dir,
)

# %%
df_of_plus, cca_column_info = apply_cca_projection(df_of, return_column_info=True)

for col_name, feats in cca_column_info.items():
    logger.info(f"{col_name}: {feats}")
    print(f"{col_name}: {feats}")

# %%
feature_list = ["cca", "cca_top3"]
plot_feature_correlations(df_of_plus, feature_list, OPTICAL_FLOW_FEATURE, output_dir)
# %%
