"""Settings for working with Timelapse Feature Explorer (TFE)."""

TFE_IMAGE_MANIFEST_NAME_MAP: dict[str, str] = {
    "CDH5": "cdh5_classic_seg_zarr",
    "grid": "grid_seg_zarr",
}
"""Map of TFE segmentation type to image manifest name."""

TFE_BACKDROP_TYPES: list[str] = ["bf_slice", "bf_std_dev", "gfp_max_proj"]
"""List of TFE backdrop types to generate."""

TFE_DEFAULT_DATASETS: list[str] = ["20250618_20X"]
"""Default dataset(s) for converting to TFE."""

TFE_DEFAULT_POSITIONS: list[int] = [0]
"""Default position(s) for converting to TFE."""
