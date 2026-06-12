from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column

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

output_path = get_output_path("updated_cdh5_classic_segmentation")
manifest = load_dataframe_manifest("cdh5_classic_segmentation")

for dataset, location in manifest.locations.items():
    print(f"Updating columns for '{dataset}'")
    df_original = load_dataframe(location, delay=True)
    df_updated = df_original.drop(columns=column_drops).rename(columns=column_renames).compute()
    file_name = f"{dataset}_cdh5_segprops.parquet"
    df_updated.to_parquet(output_path / file_name, index=False)
