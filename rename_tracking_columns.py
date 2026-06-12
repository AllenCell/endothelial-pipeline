from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column

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

output_path = get_output_path("updated_cdh5_classic_segmentation_tracking")
manifest = load_dataframe_manifest("cdh5_classic_segmentation_tracking")

for dataset, location in manifest.locations.items():
    print(f"Updating columns for '{dataset}'")
    df_original = load_dataframe(location, delay=True)
    df_updated = df_original.drop(columns=column_drops).rename(columns=column_renames).compute()
    file_name = f"{dataset}_tracking.parquet"
    df_updated.to_parquet(output_path / file_name, index=False)
