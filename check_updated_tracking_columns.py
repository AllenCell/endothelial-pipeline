from pathlib import Path

import pandas as pd

from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import DataframeLocation
from endo_pipeline.settings.column_names import ColumnName as Column

# Define renamed columns

column_drops = ["image_index"]

column_renames = {
    "area": Column.SegData.AREA_PX_SQ,
    "dataset_name": Column.DATASET,
    "eccentricity": Column.SegData.ECCENTRICITY,
    "label": Column.SegData.LABEL,
    "orientation": Column.SegData.ORIENTATION,
    "perimeter": Column.SegData.PERIMETER_PX,
    "position": Column.POSITION,
    "T": Column.TIMEPOINT,
    "touches_border": Column.SegDataFilters.IS_EDGE_SEGMENTATION,
    "track_id": Column.TRACK_ID,
    "centroid_X": "centroid_x",
    "centroid_Y": "centroid_y",
}

# Load original and demo mode dataframes

df_old = load_dataframe(DataframeLocation(fmsid="85a9cd3c990b4bd39b71e26be0de9ee3"), delay=True)
df_new = load_dataframe(
    DataframeLocation(
        path=Path(
            "/allen/aics/users/jessica.yu/2026-06-12/cdh5_tracking/20250319_20X_tracking.parquet"
        )
    )
)

# Filter original dataframe to demo mode limits and apply renamed and dropped columns

df_old_subset = df_old[(df_old["position"] < 2) & (df_old["T"] < 6)].reset_index(drop=True)
df_old_subset = df_old_subset.drop(columns=column_drops)
df_old_subset = df_old_subset.rename(columns=column_renames).compute()

# Also filter new dataframe because final 4 timepoints have fewer than 5
# entries in `matched_query_label` and `optimized_metric_value because demo
# mode only runs for 10 timepoints, and are therefore by definition not
# the same as the original data, which was run for the full duration
df_new_subset = df_new[df_new["frame_number"] < 6].reset_index(drop=True)

# Compare the two dataframes to ensure they match

print(f"Old shape: {df_old_subset.shape}")
print(f"New shape: {df_new_subset.shape}")

df_old_subset["matching_method"] = df_old_subset["matching_method"].astype("string")
df_new_subset["matching_method"] = df_new_subset["matching_method"].astype("string")
df_old_subset["dataset"] = df_old_subset["dataset"].astype("string")
df_new_subset["dataset"] = df_new_subset["dataset"].astype("string")
pd.testing.assert_frame_equal(df_old_subset, df_new_subset)
