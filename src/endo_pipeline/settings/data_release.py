S3_INTERNAL_DIRECTORY = "s3://allencell-internal-quilt/endo_stg/"
"""Path prefix for internal S3 storage of release datasets."""

SOURCE_COL = "local_zarr_path"
"""Name of the column for local zarr paths in the upload CSV."""

DEST_COL = "s3_zarr_path"
"""Name of the column for S3 zarr paths in the upload / remove CSV."""
