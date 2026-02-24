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
    rank_features_and_plot_histograms,
    run_lda_feature_ranking,
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
annotation_path = "//allen/aics/users/chantelle.leveille/annotations"
mixed_mig_files = [
    {
        "dataset_name": "20250319_20X",
        "position": 2,
        "fname": "mixed_mig/12.2 shear stress 20250319_20X_P2-annotations.csv",
    },
    {
        "dataset_name": "20250319_20X",
        "position": 3,
        "fname": "mixed_mig/12.2 shear stress 20250319_20X_P3-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 0,
        "fname": "mixed_mig/14.65 shear stress 20250813_20X_P0-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 1,
        "fname": "mixed_mig/14.65 shear stress 20250813_20X_P1-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 4,
        "fname": "mixed_mig/14.65 shear stress 20250813_20X_P4-annotations.csv",
    },
]

coherent_mig_files = [
    {
        "dataset_name": "20250319_20X",
        "position": 0,
        "fname": "coherent_mig/12.2 shear stress 20250319_20X_P0-annotations.csv",
    },
    {
        "dataset_name": "20250319_20X",
        "position": 2,
        "fname": "coherent_mig/12.2 shear stress 20250319_20X_P2-annotations.csv",
    },
    {
        "dataset_name": "20250319_20X",
        "position": 5,
        "fname": "coherent_mig/12.2 shear stress 20250319_20X_P5-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 1,
        "fname": "coherent_mig/14.65 shear stress 20250813_20X_P1-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 3,
        "fname": "coherent_mig/14.65 shear stress 20250813_20X_P3-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 5,
        "fname": "coherent_mig/14.65 shear stress 20250813_20X_P5-annotations.csv",
    },
]

# %%
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca = fit_pca(num_pcs=80)
# %%
df_mig_list = []
for file_info in mixed_mig_files + coherent_mig_files:
    dataset_name = str(file_info["dataset_name"])

    df = get_dataframe_for_dynamics_workflows(
        dataset_name, dataframe_manifest, pca=pca, filter_dataframe=False
    )

    fname = file_info["fname"]
    df_annotation = pd.read_csv(f"{annotation_path}/{fname}")
    pairs_df = df_annotation[["Track", "Frame"]]
    merged = df.merge(
        pairs_df, left_on=["crop_index", "frame_number"], right_on=["Track", "Frame"], how="inner"
    )
    merged["coherent_migration"] = file_info in coherent_mig_files
    df_mig_list.append(merged)

    if len(merged) != len(df_annotation):
        logger.error("File '%s' had different number of rows after merge", fname)
        raise ValueError(f"Different dataframe lengths: '{len(df_annotation)}' vs. '{len(merged)}'")

df_mig = pd.concat(df_mig_list, ignore_index=True)


# %% PC ranking and histogram plotting
rank_features_and_plot_histograms(
    df_mig,
    features_to_rank=pc_columns_to_keep,
    output_dir=output_dir,
    label_column="coherent_migration",
)

# %% LDA feature ranking and histogram plotting, pcs only
df_lda, df_proj, lda_csv_path = run_lda_feature_ranking(
    df_mig, pc_columns_to_keep, output_dir, "pcs_only"
)
rank_features_and_plot_histograms(
    df_proj,
    list(df_proj.columns.drop("coherent_migration")),
    output_dir=output_dir,
    label_column="coherent_migration",
    fname="find_coherent_mig_histograms_lda_pcs_only.png",
)


# %% Upload LDA feature ranking results to FMS
if UPLOAD_TO_FMS:
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
    features_in_lda_rank=df_lda["features"],
    lda_weights=df_lda["weights"],
    lda_intercept=df_lda["intercept"][0],
    sparse_axes=[2.0, 3.0, 4.0],
)
# %%
