from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm

cdh5_measured_feat_data_dir = Path(
    "//allen/aics/users/serge.parent/cellsmap/results/2026-02-27/cdh5_get_measured_features"
)
live_seg_feat_data_dir = Path(
    "//allen/aics/users/serge.parent/cellsmap/results/2026-03-03/make_seg_feats_manifest/segmentation_features_dataframes"
)
pc_diffae_classic_feat_data_dir = Path(
    "//allen/aics/users/serge.parent/cellsmap/results/2026-03-03/track_integration"
)

col_names_to_fix = {
    "cell_fluoresnce_min (a.u.)": "cell_fluorescence_min (a.u.)",
    "edge_means (a.u.)": "edge_fluorescence_means (a.u.)",
    "edge_std (a.u.)": "edge_fluorescence_std (a.u.)",
    "node_means (a.u.)": "node_fluorescence_means (a.u.)",
    "node_std (a.u.)": "node_fluorescence_std (a.u.)",
    "edge_and_node_means (a.u.)": "edge_and_node_fluorescence_means (a.u.)",
    "edge_and_node_std (a.u.)": "edge_and_node_fluorescence_std (a.u.)",
}

print("Fixing column names...")
for fp in tqdm(
    cdh5_measured_feat_data_dir.glob("*_cdh5_segprops.parquet"),
    desc="Fixing cdh5 measured feature tables...",
):
    df = pd.read_parquet(fp)
    df.rename(columns={"cell_fluoresnce_min (a.u.)": "cell_fluorescence_min (a.u.)"}, inplace=True)
    df.to_parquet(fp)
for fp in tqdm(
    live_seg_feat_data_dir.glob("*_segmentation_features.parquet"),
    desc="Fixing live segmentation feature tables...",
):
    df = pd.read_parquet(fp)
    df.rename(columns=col_names_to_fix, inplace=True)
    df.to_parquet(fp)
for fp in tqdm(
    pc_diffae_classic_feat_data_dir.glob("*_pc_diffae_seg_feats_merged.parquet"),
    desc="Fixing merged PC-DiffAE-classic tables...",
):
    df = pd.read_parquet(fp)
    df.rename(columns=col_names_to_fix, inplace=True)
    df.to_parquet(fp)

print("Checking that tables were saved with updated column names...")
for fp in tqdm(
    cdh5_measured_feat_data_dir.glob("*_cdh5_segprops.parquet"),
    desc="Checking cdh5 measured feature tables...",
):
    df = dd.read_parquet(fp)
    for old_col_name, new_col_name in col_names_to_fix.items():
        old_col_name_in_df = old_col_name in df.columns
        new_col_name_in_df = new_col_name in df.columns
        print({f"{old_col_name}: {old_col_name_in_df}, {new_col_name}: {new_col_name_in_df}"})
for fp in tqdm(
    live_seg_feat_data_dir.glob("*_segmentation_features.parquet"),
    desc="Checking live segmentation feature tables...",
):
    df = dd.read_parquet(fp)
    for old_col_name, new_col_name in col_names_to_fix.items():
        old_col_name_in_df = old_col_name in df.columns
        new_col_name_in_df = new_col_name in df.columns
        print({f"{old_col_name}: {old_col_name_in_df}, {new_col_name}: {new_col_name_in_df}"})

for fp in tqdm(
    pc_diffae_classic_feat_data_dir.glob("*_pc_diffae_seg_feats_merged.parquet"),
    desc="Checking merged PC-DiffAE-classic tables...",
):
    df = dd.read_parquet(fp)
    for old_col_name, new_col_name in col_names_to_fix.items():
        old_col_name_in_df = old_col_name in df.columns
        new_col_name_in_df = new_col_name in df.columns
        print({f"{old_col_name}: {old_col_name_in_df}, {new_col_name}: {new_col_name_in_df}"})
