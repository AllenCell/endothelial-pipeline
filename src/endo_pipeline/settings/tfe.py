"""Settings for working with Timelapse Feature Explorer (TFE)."""

TFE_IMAGE_MANIFEST_NAME_MAP: dict[str, str] = {
    "CDH5": "cdh5_classic_seg_zarr",
    "grid": "grid_seg_zarr",
}
"""Map of segmentation type to image manifest name."""

TFE_BACKDROP_TYPES: list[str] = ["bf_slice", "bf_std_dev", "gfp_max_proj"]
"""List of backdrop types to generate."""
