"""Constants used for staging manifests to S3 bucket."""

S3_STAGING_DIRECTORY: str = "s3://allencell-internal-quilt/endo_stg/"
"""Internal S3 directory."""

STAGING_SOURCE_COLUMN_NAME: str = "source"
"""Name of source column in staging CSV."""

STAGING_TARGET_COLUMN_NAME: str = "target"
"""Name of target column in staging CSV."""

STAGING_IMAGE_MANIFEST_NAMES = [
    "image_zarr",
    "cdh5_classic_seg_zarr",
    "nuclear_labelfree_seg_zarr",
    "grid_seg_zarr",
]
"""List of names of image manifests to stage."""

DATAFRAME_MANIFEST_STAGING_FOLDERS = {
    "cdh5_classic_segmentation": "vecadherin_segmentation_features/"
}
"""Mapping of dataframe manifest names to subdirectories."""

MODEL_MANIFEST_STAGING_FOLDERS = {
    "diffae_baseline_exclude_cell_piling": "diffae_baseline_exclude_cell_piling/",
    "nuc_pred_labelfree": "nuc_pred_labelfree/",
}
"""Mapping of model manifest names to subdirectories."""
