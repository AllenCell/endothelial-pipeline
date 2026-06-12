from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnNamePrefix as ColumnPrefix

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

# unfiltered

output_path = get_output_path("updated_cell_centered_features_unfiltered")
manifest = load_dataframe_manifest("cell_centered_features_unfiltered")

for dataset, location in manifest.locations.items():
    print(f"Updating columns for '{dataset}'")
    df_original = load_dataframe(location, delay=True)
    df_updated = df_original.drop(columns=column_drops).rename(columns=column_renames).compute()
    file_name = f"{dataset}_pc_diffae_seg_feats_merged.parquet"
    df_updated.to_parquet(output_path / file_name, index=False)

# filtered

output_path = get_output_path("updated_cell_centered_features_filtered")
manifest = load_dataframe_manifest("cell_centered_features_filtered")

for dataset, location in manifest.locations.items():
    print(f"Updating columns for '{dataset}'")
    df_original = load_dataframe(location, delay=True)
    df_updated = df_original.drop(columns=["zarr_path"]).rename(columns=column_renames).compute()
    file_name = f"{dataset}_pc_diffae_seg_feats_merged_filtered.parquet"
    df_updated.to_parquet(output_path / file_name, index=False)
