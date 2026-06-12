from pathlib import Path

import pandas as pd

from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import DataframeLocation
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNamePrefix as ColumnPrefix

# Define renamed columns

column_drops = [
    f"nuc_seg_intens_{metric}_{channel}"
    for metric in ["maxs", "means", "medians", "mins", "pct25s", "pct75s", "stds"]
    for channel in ["BF", "EGFP"]
]

column_renames = {
    "dataset_name": Column.DATASET,
    "T": Column.TIMEPOINT,
    "cdh5_segmentation_label": Column.SegDataWorkflowVerification.CDH5_SEGMENTATION_LABEL,
    "nuclei_segmentation_labels": Column.SegDataWorkflowVerification.NUCLEI_LABELS_IN_CDH5_SEGMENTATION,
    "nuclei_seg_in_cdh5_seg_frac": Column.SegDataWorkflowVerification.NUCLEI_FRACTION_IN_CDH5_SEGMENTATION,
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

df_old = load_dataframe(DataframeLocation(fmsid="73ecf55d8d71438fbcc83af4961b8fd1"), delay=True)
df_new = load_dataframe(
    DataframeLocation(
        path=Path(
            "/allen/aics/users/jessica.yu/2026-06-12/nuclei_measured_features/20250319_20X_nuclei_labelfree_features.parquet"
        )
    )
)

# Filter original dataframe to demo mode limits and apply renamed and dropped columns

df_old_subset = df_old[(df_old["position"] < 2) & (df_old["T"] < 3)].reset_index(drop=True)
df_old_subset = df_old_subset.drop(columns=column_drops)
df_old_subset = df_old_subset.rename(columns=column_renames).compute().dropna(axis=1, how="all")

# Compare the two dataframes to ensure they match

print(f"Old shape: {df_old_subset.shape}")
print(f"New shape: {df_new.shape}")

df_old_subset["dataset"] = df_old_subset["dataset"].astype("string")
df_new["dataset"] = df_new["dataset"].astype("string")
pd.testing.assert_frame_equal(df_old_subset, df_new)
