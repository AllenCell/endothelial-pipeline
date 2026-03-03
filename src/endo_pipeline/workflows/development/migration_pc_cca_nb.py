# %%
import logging

import pandas as pd

from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_subset_of_timepoint_annotations,
    load_dataset_config,
)
from endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    filter_dataframe_by_annotations,
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    project_features_to_pcs,
)
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.analyze.migration_pc.cca_analysis import (
    calculate_cca_results,
    plot_cca_projection_validation,
    plot_cca_results,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES, SEGMENTATION_FEATURE_COLUMNS
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)

DESCRIPTION = "Optical flow on BF for migration coherence metric; CCA ranks top contributing PCs."

OPTICAL_FLOW_FEATURE = "optical_flow_angle_std_dt1"  # "optical_flow_mean_unit_vector_dt1"
CLASSIC_FEATURE = "vec_mean_mag_in_crop"
PLOT_CLASSIC = False
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
# print a warning if df_pca.shape is not the same as df_of.shape
if df_pca.shape[0] != df_of.shape[0]:
    logger.warning(
        f"Warning: PCA dataframe has {df_pca.shape[0]} rows. "
        f"Optical Flow dataframe has {df_of.shape[0]} rows. "
        "The downstream cca wieghts need to be scaled. "
        "This will be resolved when we have optical flow features for all timepoints. "
    )

# %%
df_of_sub = df_of.dropna(subset=DIFFAE_PC_COLUMN_NAMES[:80] + [OPTICAL_FLOW_FEATURE])
cca_results, cca_csv_path = calculate_cca_results(
    df_of_sub,
    OPTICAL_FLOW_FEATURE,
    input_features=DIFFAE_PC_COLUMN_NAMES[:80],
    output_dir=output_dir,
    scale_cca=False,
)
plot_cca_results(cca_results, output_dir)

# %%
plot_cca_projection_validation(
    df_of_sub,
    cca_results,
    output_dir,
)

# %%
UPLOAD_TO_FMS = True
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
if PLOT_CLASSIC:

    # Classic feature has a harder time capturing the manual annotations of migratory groups, and
    # the CCA does not correlate as highly when using it either.

    dataframe_manifest_classic = load_dataframe_manifest("test_live_merged_seg_features")
    dataframe_cell_centric_diffae = load_dataframe_manifest("pc_diffae_tracked_seg_features")

    df_classic_list = []
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name} for classic features")
        dataframe_location_classic = get_dataframe_location_for_dataset(
            dataframe_manifest_classic, dataset_name
        )
        df_classic_delayed = load_dataframe(dataframe_location_classic, delay=True)

        dataframe_location_cell_centered = get_dataframe_location_for_dataset(
            dataframe_cell_centric_diffae, dataset_name
        )
        df_cell_centered = load_dataframe(dataframe_location_cell_centered)

        cols_to_compute = list(
            set(
                SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"]
                + SEGMENTATION_FEATURE_COLUMNS["filters"]
            )
        )
        df_classic = df_classic_delayed[cols_to_compute].compute()
        df_classic = df_classic[df_classic.is_included]
        df_classic = calculate_derived_data_dynamics_dependent(
            df_classic, compute_per_crop_metrics=True
        )

        df_merged = pd.merge(
            df_classic,
            df_cell_centered,
            left_on=["dataset_name", "position", "T", "label", "track_id"],
            right_on=["dataset_name", "position", "frame_number", "label", "track_id"],
            how="inner",
        )

        timepoint_annotations = get_subset_of_timepoint_annotations(annotations_to_ignore=[])
        df_filtered = filter_dataframe_by_annotations(
            df_merged,
            load_dataset_config(dataset_name),
            timepoint_annotations=timepoint_annotations,
        )

        df_classic_list.append(df_filtered)

    df_classic = pd.concat(df_classic_list, ignore_index=True)
    df_classic_list = None  # clear classic list to save memory
