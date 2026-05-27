"""Constants used for staging manifests to S3 bucket."""

S3_STAGING_DIRECTORY: str = "s3://allencell-internal-quilt/endo_stg/"
"""Internal S3 directory."""

STAGING_SOURCE_COLUMN_NAME: str = "local_path_staging"
"""Name of source column in staging CSV."""

STAGING_TARGET_COLUMN_NAME: str = "s3_uri_staging"
"""Name of target column in staging CSV."""

IMAGE_MANIFEST_STAGING_FOLDERS = {
    "image_zarr": "",
    "cdh5_classic_seg_zarr": "vecadherin_segmentation_zarrs/",
    "nuclear_labelfree_seg_zarr": "nuclear_segmentation_zarrs/",
}
"""Mapping of image manifest names to subdirectories."""
