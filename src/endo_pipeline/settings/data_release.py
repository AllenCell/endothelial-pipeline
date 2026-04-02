S3_INTERNAL_DIRECTORY: str = "s3://allencell-internal-quilt/endo_stg/"
"""Path prefix for internal S3 storage of release datasets."""

SOURCE_COL: str = "local_zarr_path"
"""Name of the column for local zarr paths in the upload CSV."""

DEST_COL: str = "s3_zarr_path"
"""Name of the column for S3 zarr paths in the upload / remove CSV."""

BFF_FILE_PATH_COL: str = "File Path"
"""Name of the required column for file paths in the BFF CSV."""

DEST_NUC_SEG_DIR: str = "nuclear_segmentation_zarrs/"
"""S3 directory for nuclear segmentation zarrs."""

DEST_CDH5_SEG_DIR: str = "vecadherin_segmentation_zarrs/"
"""S3 directory for VE-cadherin segmentation zarrs."""
