# %%
import logging

import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.migration_pc.lda_analysis import (
    apply_lda_projection,
    build_lda_outputs,
    compute_separation_power,
    fit_lda_feature_ranking,
    plot_lda_optimal_axis,
    plot_ranked_feature_histograms,
)
from endo_pipeline.library.analyze.migration_pc.specify_manual_annotations import (
    ANNOTATION_PATH,
    COHERENT_MIG_FILES,
    COHERENT_MIGRATION_COL,
    MIXED_MIG_FILES,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)

DESCRIPTION = "Manual annotations for migration type; LDA ranks top contributing PCs."

UPLOAD_TO_FMS = False

# %%
output_dir = get_output_path("find_coherent_mig")
pc_columns_to_keep = DIFFAE_PC_COLUMN_NAMES[:80]

# %%
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca = fit_pca(num_pcs=80)
# %%
df_mig_list = []
for file_info in MIXED_MIG_FILES + COHERENT_MIG_FILES:
    dataset_name = str(file_info["dataset_name"])

    df = get_dataframe_for_dynamics_workflows(
        dataset_name, dataframe_manifest, pca=pca, filter_dataframe=False
    )

    fname = file_info["fname"]
    df_annotation = pd.read_csv(f"{ANNOTATION_PATH}/{fname}")
    df_annotation["crop_index"] = df_annotation["Track"] - 1

    pairs_df = df_annotation[["crop_index", "Frame"]]
    merged = df.merge(
        pairs_df,
        left_on=["crop_index", "frame_number"],
        right_on=["crop_index", "Frame"],
        how="inner",
    )
    merged[COHERENT_MIGRATION_COL] = file_info in COHERENT_MIG_FILES
    merged["migration_type"] = "coherent" if file_info in COHERENT_MIG_FILES else "mixed"
    df_mig_list.append(merged)

    if len(merged) != len(df_annotation):
        logger.error("File '%s' had different number of rows after merge", fname)
        raise ValueError(f"Different dataframe lengths: '{len(df_annotation)}' vs. '{len(merged)}'")

df_mig = pd.concat(df_mig_list, ignore_index=True)


# %% PC ranking and histogram plotting
pc_ranking = compute_separation_power(df_mig[pc_columns_to_keep], df_mig["migration_type"])
plot_ranked_feature_histograms(
    df_mig,
    pc_ranking,
    output_dir=output_dir,
    label_column="migration_type",
)

# %% LDA feature ranking and histogram plotting, pcs only
features_ranked, optimal_axis, lda_intercept, projected_data = fit_lda_feature_ranking(
    df_mig, pc_columns_to_keep, binary_target_feature=COHERENT_MIGRATION_COL
)
plot_lda_optimal_axis(features_ranked, optimal_axis, output_dir, "pcs_only")
df_lda, df_proj, lda_csv_path = build_lda_outputs(
    df_mig,
    features_ranked,
    optimal_axis,
    lda_intercept,
    projected_data,
    binary_target_feature=COHERENT_MIGRATION_COL,
    output_dir=output_dir,
    fname_suffix="pcs_only",
)
lda_features = list(df_proj.columns.drop([COHERENT_MIGRATION_COL]))
lda_ranking = compute_separation_power(df_proj[lda_features], df_proj[COHERENT_MIGRATION_COL])
plot_ranked_feature_histograms(
    df_proj,
    lda_ranking,
    output_dir=output_dir,
    label_column=COHERENT_MIGRATION_COL,
    fname="find_coherent_mig_histograms_lda_pcs_only.png",
)

# %% Run LDA on randomized labels as a control
df_mig_random = df_mig.copy()
df_mig_random[COHERENT_MIGRATION_COL] = df_mig[COHERENT_MIGRATION_COL].sample(frac=1).values
features_ranked_random, optimal_axis_random, lda_intercept_random, projected_data_random = (
    fit_lda_feature_ranking(
        df_mig_random,
        pc_columns_to_keep,
        binary_target_feature=COHERENT_MIGRATION_COL,
    )
)
# %%
plot_lda_optimal_axis(
    features_ranked_random,
    optimal_axis_random,
    output_dir,
    "pcs_only_random",
    title_suffix="* scrambled annotations",
)
# %%
df_lda_random, df_proj_random, _ = build_lda_outputs(
    df_mig_random,
    features_ranked_random,
    optimal_axis_random,
    lda_intercept_random,
    projected_data_random,
    binary_target_feature=COHERENT_MIGRATION_COL,
    minimal_weight=None,
    output_dir=output_dir,
    fname_suffix="pcs_only_random",
)
lda_random_features = list(df_proj_random.columns.drop([COHERENT_MIGRATION_COL]))
lda_random_ranking = compute_separation_power(
    df_proj_random[lda_random_features], df_proj_random[COHERENT_MIGRATION_COL]
)
# %%
plot_ranked_feature_histograms(
    df_proj_random,
    lda_random_ranking,
    output_dir=output_dir,
    label_column=COHERENT_MIGRATION_COL,
    fname="find_coherent_mig_histograms_lda_pcs_only_random.png",
    legend_suffix="* scrambled",
)

# %% Upload LDA feature ranking results to FMS
if UPLOAD_TO_FMS:
    if lda_csv_path is None:
        raise ValueError("lda_csv_path is None; ensure build_lda_outputs is called with save=True")
    dataset_config = load_dataset_config("20250319_20X")
    dataset_config_2 = load_dataset_config("20250813_20X")

    annotations = build_fms_annotations(
        dataset=[dataset_config, dataset_config_2], additional_notes=DESCRIPTION
    )
    upload_file_to_fms(lda_csv_path, annotations=annotations, file_type="csv")


# %% Test applying LDA projection to an original dataframe
dataset_name = "20250319_20X"
df = get_dataframe_for_dynamics_workflows(
    dataset_name, dataframe_manifest, pca=pca, filter_dataframe=True
)
# %% Use saved lda weights to apply LDA projection to original dataframe
lda_dataframe_manifest = load_dataframe_manifest("lda_weights")
lda_location = get_dataframe_location_for_dataset(lda_dataframe_manifest, "80_pcs")
df_lda = load_dataframe(lda_location)
# %%
df_proj_full = apply_lda_projection(
    df,
    features_in_lda_rank=df_lda["features"].to_list(),
    lda_weights=df_lda["weights"].to_numpy(),
    lda_intercept=df_lda["intercept"][0],
    sparse_axes=[2.0, 3.0, 4.0, 5.0],
)
# %%
