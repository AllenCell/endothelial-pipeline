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

# Load original and demo mode dataframes

df_old = load_dataframe(DataframeLocation(fmsid="299ccaacef4f4fc695a2946b3fb02b2d"), delay=True)
df_new = load_dataframe(
    DataframeLocation(
        path=Path(
            "/allen/aics/users/jessica.yu/2026-06-12/merged_segmentation_features/20250319_20X_live_segmentation_features.parquet"
        )
    ),
    delay=True,
)

# Update original dataframe with renamed and dropped columns; sort all columns

df_old_updated = (
    df_old.drop(columns=column_drops).rename(columns=column_renames).compute().sort_index(axis=1)
)
df_new_sorted = df_new.compute().sort_index(axis=1)

# Compare the two dataframes to ensure they match

print(f"Old shape: {df_old_updated.shape}")
print(f"New shape: {df_new_sorted.shape}")

df_old_updated["dataset"] = df_old_updated["dataset"].astype("string")
df_new_sorted["dataset"] = df_new_sorted["dataset"].astype("string")
pd.testing.assert_frame_equal(df_old_updated, df_new_sorted)
