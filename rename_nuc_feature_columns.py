from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNamePrefix as ColumnPrefix

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

output_path = get_output_path("updated_nuclei_labelfree_segmentation")
manifest = load_dataframe_manifest("nuclei_labelfree_segmentation")

for dataset, location in manifest.locations.items():
    print(f"Updating columns for '{dataset}'")
    df_original = load_dataframe(location, delay=True)
    df_updated = df_original.drop(columns=column_drops).rename(columns=column_renames).compute()
    file_name = f"{dataset}_nuclei_labelfree_features.parquet"
    df_updated.to_parquet(output_path / file_name, index=False)
