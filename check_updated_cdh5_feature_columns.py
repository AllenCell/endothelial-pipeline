from pathlib import Path

import pandas as pd

from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import DataframeLocation
from endo_pipeline.settings.column_names import ColumnName as Column

# Define renamed columns

column_drops = ["filepath_raw_image", "filepath_segmentation_image"]

column_renames = {
    "dataset_name": Column.DATASET,
    "position": Column.POSITION,
    "T": Column.TIMEPOINT,
    "cell_label": Column.SegData.LABEL,
    "cell_centroid": Column.SegData.CENTROID,
    "cell_area (px**2)": Column.SegData.AREA_PX_SQ,
    "cell_perimeter (px)": Column.SegData.PERIMETER_PX,
    "cell_solidity": Column.SegData.SOLIDITY,
    "major_axis_length": Column.SegData.MAJOR_AXIS,
    "minor_axis_length": Column.SegData.MINOR_AXIS,
    "cell_eccentricity": Column.SegData.ECCENTRICITY,
    "cell_orientation": Column.SegData.ORIENTATION,
    "cell_fluorescence_mean (a.u.)": Column.SegData.CELL_FLUOR_MEAN,
    "cell_fluorescence_std (a.u.)": Column.SegData.CELL_FLUOR_STD,
    "cell_fluorescence_median (a.u.)": Column.SegData.CELL_FLUOR_MEDIAN,
    "cell_fluorescence_min (a.u.)": Column.SegData.CELL_FLUOR_MIN,
    "cell_fluorescence_pct25 (a.u.)": Column.SegData.CELL_FLUOR_PCT25,
    "cell_fluorescence_pct75 (a.u.)": Column.SegData.CELL_FLUOR_PCT75,
    "cell_fluorescence_max (a.u.)": Column.SegData.CELL_FLUOR_MAX,
    "neighboring_cell_labels": Column.SegData.NEIGHBOR_LABELS,
    "edge_labels": Column.SegDataWorkflowVerification.EDGE_LABELS,
    "node_labels": Column.SegDataWorkflowVerification.NODE_LABELS,
    "node_pair_labels": Column.SegDataWorkflowVerification.NODE_PAIR_LABELS,
    "edge_fluorescences (a.u.)": Column.SegData.EDGE_FLUOR,
    "node_fluorescences (a.u.)": Column.SegData.NODE_FLUOR,
    "touches_image_border": Column.SegDataFilters.IS_EDGE_SEGMENTATION,
}

# Load original and demo mode dataframes

df_old = load_dataframe(DataframeLocation(fmsid="1f6f4e7206774e32a1e79dd6fb308414"), delay=True)
df_new = load_dataframe(
    DataframeLocation(
        path=Path(
            "/allen/aics/users/jessica.yu/2026-06-11/cdh5_measured_features/20250319_20X_cdh5_segprops.parquet"
        )
    )
)

# Filter original dataframe to demo mode limits and apply renamed and dropped columns

df_old_subset = df_old[(df_old["position"] < 2) & (df_old["T"] < 3)].reset_index(drop=True)
df_old_subset = df_old_subset.drop(columns=column_drops)
df_old_subset = df_old_subset.rename(columns=column_renames).compute()

# Compare the two dataframes to ensure they match

print(f"Old shape: {df_old_subset.shape}")
print(f"New shape: {df_new.shape}")

df_old_subset["dataset"] = df_old_subset["dataset"].astype("string")
df_new["dataset"] = df_new["dataset"].astype("string")
pd.testing.assert_frame_equal(df_old_subset, df_new)
