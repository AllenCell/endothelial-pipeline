"""Calculate 5th and 95th percentiles of key features across all shear stress datasets.

Uses the grid_based_features_filtered manifest to load data for each dataset
in the "shear_stress" collection and computes percentiles for:
- polar_r
- rho (PC3 flipped)
- migration coherence (EMA01 unit vector mean)
- migration speed (mean speed)
"""

# %%
import logging

import pandas as pd

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.workflow_defaults import GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME

logger = logging.getLogger(__name__)
# %%
output_dir = get_output_path("feature_percentiles")

# Features to compute percentiles for
FEATURES = {
    Column.DiffAEData.POLAR_RADIUS: "polar_r",
    Column.DiffAEData.PC3_FLIPPED: "rho",
    Column.OpticalFlow.UNIT_VECTOR_MEAN: "migration_coherence",
    Column.OpticalFlow.SPEED_MEAN: "migration_speed",
}

# Columns needed from the grid-filtered manifest (non-optical-flow)
BASE_COLUMNS = [
    Column.DATASET,
    Column.POSITION,
    Column.TIMEPOINT,
    Column.DiffAEData.START_X,
    Column.DiffAEData.START_Y,
    Column.DiffAEData.POLAR_RADIUS,
    Column.DiffAEData.PC3_FLIPPED,
]

# %%
dataset_names = get_datasets_in_collection("shear_stress", "perturbation")
manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)

all_dfs: list[pd.DataFrame] = []
for dataset_name in dataset_names:
    if dataset_name not in manifest.locations:
        logger.warning("Skipping %s: not in manifest", dataset_name)
        continue
    logger.info("Loading %s...", dataset_name)
    df = load_dataframe(manifest.locations[dataset_name], delay=True)[BASE_COLUMNS].compute()
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

# Merge optical flow features
logger.info("Merging optical flow features...")
df_all = add_optical_flow_features(df_all, datasets=dataset_names)

# %% Compute percentiles per dataset and overall
feature_cols = list(FEATURES.keys())
results: list[dict] = []

# Per-dataset percentiles
for dataset_name, df_ds in df_all.groupby(Column.DATASET):
    for col, label in FEATURES.items():
        if col not in df_ds.columns:
            continue
        data = df_ds[col].dropna()
        if len(data) == 0:
            continue
        results.append(
            {
                "dataset": dataset_name,
                "feature": label,
                "column_name": col,
                "p5": data.quantile(0.05),
                "p95": data.quantile(0.95),
                "n": len(data),
            }
        )

# Overall (pooled) percentiles
for col, label in FEATURES.items():
    if col not in df_all.columns:
        continue
    data = df_all[col].dropna()
    if len(data) == 0:
        continue
    results.append(
        {
            "dataset": "ALL_POOLED",
            "feature": label,
            "column_name": col,
            "p5": data.quantile(0.05),
            "p95": data.quantile(0.95),
            "n": len(data),
        }
    )

df_results = pd.DataFrame(results)
df_results["p5"] = df_results["p5"].round(1)
df_results["p95"] = df_results["p95"].round(1)
df_pooled = df_results[df_results["dataset"] == "ALL_POOLED"]
print(df_pooled.to_string(index=False))

# Save to CSV
output_path = output_dir / "feature_percentiles_shear_stress_and_perturbation.csv"
df_results.to_csv(output_path, index=False)
logger.info("Saved results to %s", output_path)

# %%
