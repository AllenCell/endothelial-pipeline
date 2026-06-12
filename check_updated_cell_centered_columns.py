from pathlib import Path

import pandas as pd

from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import DataframeLocation
from endo_pipeline.settings.column_names import ColumnNamePrefix as ColumnPrefix

# Define renamed columns

column_drops = [
    f"nuc_seg_intens_{metric}_{channel}"
    for metric in ["maxs", "means", "medians", "mins", "pct25s", "pct75s", "stds"]
    for channel in ["BF", "EGFP"]
]
column_drops = [*column_drops, "filepath_segmentation_image", "zarr_path"]

column_renames = {
    **{
        f"nuclei_seg_with_most_overlap_{i}": f"{ColumnPrefix.NUCLEI_WITH_MOST_OVERLAP}{i}"
        for i in range(10)
    },
    **{
        f"nuc_with_most_overlap_{i}_centroid_X": f"{ColumnPrefix.NUCLEI_WITH_MOST_OVERLAP}{i}_centroid_X"
        for i in range(10)
    },
    **{
        f"nuc_with_most_overlap_{i}_centroid_Y": f"{ColumnPrefix.NUCLEI_WITH_MOST_OVERLAP}{i}_centroid_Y"
        for i in range(10)
    },
}

# Load original and demo mode dataframes for filtered and unfiltered

df_old_unfiltered = load_dataframe(DataframeLocation(fmsid="8033a1c43b914395bf604cc585b6a3ee"))
df_new_unfiltered = load_dataframe(
    DataframeLocation(
        path=Path(
            "/allen/aics/users/jessica.yu/2026-06-12/cell_centered_features/20250319_20X_pc_diffae_seg_feats_merged.parquet"
        )
    )
)

df_old_filtered = load_dataframe(DataframeLocation(fmsid="5a66fb9742054e3a991cf08f82973bda"))
df_new_filtered = load_dataframe(
    DataframeLocation(
        path=Path(
            "/allen/aics/users/jessica.yu/2026-06-12/cell_centered_features/20250319_20X_pc_diffae_seg_feats_merged_filtered.parquet"
        )
    )
)

# Apply renamed columns

df_old_unfiltered_sorted = (
    df_old_unfiltered.drop(columns=column_drops).rename(columns=column_renames).sort_index(axis=1)
)
df_old_filtered_sorted = (
    df_old_filtered.drop(columns=["zarr_path"]).rename(columns=column_renames).sort_index(axis=1)
)

df_new_unfiltered_sorted = df_new_unfiltered.sort_index(axis=1)
df_new_filtered_sorted = df_new_filtered.sort_index(axis=1)

# Compare the unfiltered two dataframes to ensure they match

print(f"Old shape: {df_old_unfiltered_sorted.shape}")
print(f"New shape: {df_new_unfiltered_sorted.shape}")

df_old_unfiltered_sorted["matching_method"] = df_old_unfiltered_sorted["matching_method"].astype(
    "string"
)
df_new_unfiltered_sorted["matching_method"] = df_new_unfiltered_sorted["matching_method"].astype(
    "string"
)
df_old_unfiltered_sorted["shear_stress_regime"] = df_old_unfiltered_sorted[
    "shear_stress_regime"
].astype("string")
df_new_unfiltered_sorted["shear_stress_regime"] = df_new_unfiltered_sorted[
    "shear_stress_regime"
].astype("string")
pd.testing.assert_frame_equal(
    df_old_unfiltered_sorted.head(10000), df_new_unfiltered_sorted.head(10000)
)

# Compare the filtered two dataframes to ensure they match

print(f"Old shape: {df_old_filtered_sorted.shape}")
print(f"New shape: {df_new_filtered_sorted.shape}")

df_old_filtered_sorted["shear_stress_regime"] = df_old_filtered_sorted[
    "shear_stress_regime"
].astype("string")
df_new_filtered_sorted["shear_stress_regime"] = df_new_filtered_sorted[
    "shear_stress_regime"
].astype("string")
pd.testing.assert_frame_equal(df_old_filtered_sorted, df_new_filtered_sorted)
