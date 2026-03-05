# %%
import logging
import re

import matplotlib.pyplot as plt
import pandas as pd

from endo_pipeline.configs import get_subset_of_timepoint_annotations, load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    filter_dataframe_by_annotations,
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.analyze.migration_pc.compare_feats import plot_lda_vs_optical_flow
from endo_pipeline.library.analyze.migration_pc.lda_analysis import apply_lda_projection
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings import SEGMENTATION_FEATURE_COLUMNS
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)

CLIP_QUANTILES = (0.01, 0.99)
COHERENT_MIGRATION_COL = "coherent_migration"

# %%
datasets = [
    "20250618_20X",
    "20250428_20X",
    "20250319_20X",
    "20250813_20X",
    "20250611_20X",
    "20250818_20X",
]

# %% Load lda weights to apply LDA projection to original dataframe
lda_dataframe_manifest = load_dataframe_manifest("lda_weights")
lda_location = get_dataframe_location_for_dataset(lda_dataframe_manifest, "80_pcs")
df_lda = load_dataframe(lda_location)
lda_features = df_lda["features"].to_list()
lda_weights = df_lda["weights"].to_numpy()
lda_intercept = float(df_lda["intercept"].iloc[0])

# %% Load diffae features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca = fit_pca(num_pcs=80)

# %% Load optical flow features
dataframe_manifest_optical_flow_new = load_dataframe_manifest("optical_flow_bf")

df_proj_full_list_new = []

for dataset_name in datasets:
    logging.info(f"Processing dataset: {dataset_name}")
    # Get PCS and LDA features
    df_dataset = get_dataframe_for_dynamics_workflows(
        dataset_name, dataframe_manifest, pca=pca, filter_dataframe=True
    )

    df_proj_full = apply_lda_projection(
        df_dataset,
        features_in_lda_rank=lda_features,
        lda_weights=lda_weights,
        lda_intercept=lda_intercept,
        sparse_axes=[2.0, 3.0, 4.0, 5.0],
    )

    # Get optical flow features
    optical_flow_location_new = get_dataframe_location_for_dataset(
        dataframe_manifest_optical_flow_new, dataset_name
    )
    df_optical_flow_new = load_dataframe(optical_flow_location_new)

    # merge the two dataframes on the dataset, position, frame_number, start_x, start_y columns
    df_proj_full_new = df_proj_full.merge(
        df_optical_flow_new,
        on=["dataset", "position", "frame_number", "start_x", "start_y"],
        how="inner",
        suffixes=("", "_optical_flow_new"),
    )

    df_proj_full_list_new.append(df_proj_full_new)

# %%
df_new = pd.concat(df_proj_full_list_new, ignore_index=True)

# %%
features = ["LDA", "LDA_SP_2", "LDA_SP_3", "LDA_SP_4", "LDA_SP_5", "pc_1", "pc_2", "pc_3"]
datasets_used = ["20250618_20X", "20250428_20X", "20250319_20X", "20250813_20X", "20250611_20X"]

# %%
dt = 1
optical_flow_features = [
    f"optical_flow_mean_speed_dt{dt}",
    f"optical_flow_mean_unit_vector_dt{dt}",
    f"optical_flow_std_speed_dt{dt}",
    f"optical_flow_mean_angle_dt{dt}",
    f"optical_flow_angle_std_dt{dt}",
    f"optical_flow_mean_u_dt{dt}",
    f"optical_flow_mean_v_dt{dt}",
    f"optical_flow_std_u_dt{dt}",
    f"optical_flow_std_v_dt{dt}",
]

df_sub_new = df_new[df_new["dataset"].isin(datasets_used)]
# %%
plot_lda_vs_optical_flow(
    df_sub_new,
    features,
    optical_flow_features,
    color_by_dataset=False,
    clip_quantiles=CLIP_QUANTILES,
)

# %% Load classic features
dataframe_manifest_classic = load_dataframe_manifest("test_live_merged_seg_features")
dataframe_cell_centric_diffae = load_dataframe_manifest("pc_diffae_tracked_seg_features")

# %%
df_list = []
for dataset_name in datasets:
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

    df_proj_full = apply_lda_projection(
        df_filtered,
        features_in_lda_rank=lda_features,
        lda_weights=lda_weights,
        lda_intercept=lda_intercept,
        sparse_axes=[2.0, 3.0, 4.0, 5.0],
    )

    df_list.append(df_proj_full)

df_all_classic = pd.concat(df_list, ignore_index=True)

# %%
classic_features = ["vec_mean_angle_in_crop", "vec_mean_mag_in_crop"]

datasets_used = ["20250611_20X", "20250618_20X"]

plot_lda_vs_optical_flow(
    df_all_classic[df_all_classic["dataset_name"].isin(datasets_used)],
    features,
    classic_features,
    color_by_dataset=False,
    figsize=(24, 2.5 * 2),
    clip_quantiles=CLIP_QUANTILES,
)

# %%
plot_lda_vs_optical_flow(
    df_all_classic,
    features,
    classic_features,
    color_by_dataset=False,
    figsize=(24, 2.5 * 2),
    clip_quantiles=CLIP_QUANTILES,
)
# %%
df_mig_list = []
annotation_manifests = [
    load_dataframe_manifest("coherent_migration_annotations"),
    load_dataframe_manifest("incoherent_migration_annotations"),
]
for annotation_manifest in annotation_manifests:
    for location_name, location in annotation_manifest.locations.items():
        parsed = re.match(r"^(coherent_mig|incoherent_mig)_(.+)_P(\d+)$", location_name)
        if parsed is None:
            logger.error("Invalid annotation location key in manifest: %s", location_name)
            raise ValueError(f"Invalid annotation location key: {location_name}")

        migration_type = parsed.group(1)
        dataset_name = parsed.group(2)
        position = int(parsed.group(3))

        if location.path is None:
            logger.error("Missing path for annotation location: %s", location_name)
            raise ValueError(f"Missing path for annotation location: {location_name}")

        df = get_dataframe_for_dynamics_workflows(
            dataset_name, dataframe_manifest, pca=pca, filter_dataframe=False
        )
        df = df[df["position"] == position]

        annotation_path = str(location.path)
        df_annotation = pd.read_csv(annotation_path)
        df_annotation["crop_index"] = df_annotation["Track"] - 1

        pairs_df = df_annotation[["crop_index", "Frame"]]
        merged = df.merge(
            pairs_df,
            left_on=["crop_index", "frame_number"],
            right_on=["crop_index", "Frame"],
            how="inner",
        )
        merged[COHERENT_MIGRATION_COL] = migration_type == "coherent_mig"
        merged["migration_type"] = "coherent" if migration_type == "coherent_mig" else "incoherent"
        df_mig_list.append(merged)

        if len(merged) != len(df_annotation):
            logger.error("File '%s' had different number of rows after merge", annotation_path)
            raise ValueError(
                f"Different dataframe lengths for '{location_name}': "
                f"'{len(df_annotation)}' vs. '{len(merged)}'"
            )

df_mig = pd.concat(df_mig_list, ignore_index=True)

df_mig_of = pd.merge(
    df_mig,
    df_new,
    left_on=["dataset", "position", "frame_number", "start_x", "start_y"],
    right_on=["dataset", "position", "frame_number", "start_x", "start_y"],
    how="inner",
)


# %%
for feat in optical_flow_features:
    is_coherent_migration = df_mig_of[COHERENT_MIGRATION_COL]
    true_vals = df_mig_of[is_coherent_migration][feat]
    false_vals = df_mig_of[~is_coherent_migration][feat]

    plt.figure(figsize=(8, 5))
    plt.hist(true_vals, bins=30, alpha=0.6, label="Coherent Migration", color="blue", density=True)
    plt.hist(false_vals, bins=30, alpha=0.6, label="Mixed Migration", color="orange", density=True)

    plt.xlabel(feat)
    plt.ylabel("Density")
    plt.title("Distribution by Coherent Migration")
    plt.legend()
    plt.show()

# %%
