S3_INTERNAL_DIRECTORY: str = "s3://allencell-internal-quilt/endo_stg/"
"""Path prefix for internal S3 storage of release datasets."""

SOURCE_COL: str = "local_zarr_path"
"""Name of the column for local zarr paths in the upload CSV."""

DEST_COL: str = "s3_zarr_path"
"""Name of the column for S3 zarr paths in the upload / remove CSV."""
