"""Settings for working with Timelapse Feature Explorer (TFE)."""

from colorizer_data import FeatureType

from endo_pipeline.settings.column_metadata import ColumnType
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE

TFE_IMAGE_MANIFEST_NAME_MAP: dict[str, str] = {
    "CDH5": "cdh5_classic_seg_zarr",
    "grid": "grid_seg_zarr",
}
"""Map of TFE segmentation type to image manifest name."""

TFE_BACKDROP_TYPES: list[str] = ["bf_slice", "bf_std_dev", "gfp_max_proj"]
"""List of TFE backdrop types to generate."""

TFE_TYPE_MAPPING: dict[ColumnType, FeatureType] = {
    ColumnType.CONTINUOUS: FeatureType.CONTINUOUS,
    ColumnType.DISCRETE: FeatureType.DISCRETE,
    ColumnType.BOOLEAN: FeatureType.CATEGORICAL,
}
"""Mapping from pipeline feature types to TFE feature types."""

TFE_REQUIRED_COLUMNS: list[str | Column] = [
    Column.DATASET,
    Column.TRACK_ID,
    Column.POSITION,
    Column.TIMEPOINT,
    Column.SegData.LABEL,
    Column.SegData.LABELS_IN_CROP,
    Column.SegDataFilters.IS_INCLUDED,
    Column.PIXEL_SIZE_XY_IN_UM,
    Column.SegData.CENTROID_X,
    Column.SegData.CENTROID_Y,
    Column.SegData.TIME_MINS,
    Column.TIME_RESOLUTION_MINUTES,
    Column.SegData.ALIGNMENT_DEG,
    Column.SegData.NUCLEI_POSITION_ANGLE_DEG,
    Column.SegData.NUCLEI_POSITION_X,
    Column.SegData.NUCLEI_POSITION_Y,
    Column.SegData.CELL_FLUOR_MEAN,
    Column.SegData.EDGE_FLUOR_MEAN,
    Column.SegData.NODE_FLUOR_MEAN,
    Column.SegData.NUM_NUCLEI_IN_CROP,
]
"""List of columns required for calculating filtering-dependent track-based features."""

TFE_FEATURES: list[ColumnNameType] = [
    # General information ======================================================
    Column.SegData.TIME_HRS,
    Column.SegData.TIME_MINS,
    Column.TRACK_ID,
    Column.TRACK_LENGTH,
    # Classic segmentation features ============================================
    Column.SegData.ALIGNMENT,
    Column.SegData.ALIGNMENT_DEG,
    Column.SegData.ORIENTATION,
    Column.SegData.ORIENTATION_DEG,
    Column.SegData.AREA_UM_SQ,
    Column.SegData.ASPECT_RATIO,
    Column.SegData.PERIMETER_UM,
    Column.SegData.ECCENTRICITY,
    Column.SegData.CELL_FLUOR_MAX,
    Column.SegData.CELL_FLUOR_MEAN,
    Column.SegData.CELL_FLUOR_MEDIAN,
    Column.SegData.EDGE_FLUOR_MEAN,
    Column.SegData.NODE_FLUOR_MEAN,
    Column.SegData.SOLIDITY,
    Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF,
    Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK,
    Column.SegData.NUM_NEIGHBORS,
    Column.SegData.NUM_NUCLEI_IN_CROP,
    # Dynamic features =========================================================
    Column.SegData.CENTROID_VELOCITY_ANGLE_DEG,
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN,
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG,
    Column.SegData.NUCLEI_POSITION_ANGLE_DEG,
    Column.SegData.VECTOR_MEAN_FOR_CROP_MAGNITUDE,
    # DiffAE-based features ====================================================
    *[ColumnTemplate.PCA_FEATURE % i for i in range(1, MAX_PCS_TO_COMPUTE + 1)],
    Column.DiffAEData.POLAR_ANGLE,
    Column.DiffAEData.POLAR_RADIUS,
    Column.DiffAEData.PC3_FLIPPED,
    # Segmentation filters =====================================================
    Column.SegDataFilters.IS_EDGE_SEGMENTATION,
    Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE,
    Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION,
    Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK,
    Column.SegDataFilters.IS_INCLUDED,
    Column.SegDataFilters.IS_VALID_BBOX,
    # Timepoint annotations ====================================================
    Column.Annotations.AUTO_BF_SCOPE_ERROR,
    Column.Annotations.AUTO_BF_TEMP_ARTIFACT,
    Column.Annotations.AUTO_GFP_SCOPE_ERROR,
    Column.Annotations.BF_SCOPE_ERROR,
    Column.Annotations.BF_TEMP_ARTIFACT,
    Column.Annotations.GFP_SCOPE_ERROR,
    Column.Annotations.CELL_PILING,
    Column.Annotations.NOT_STEADY_STATE,
    Column.Annotations.UNFED,
    Column.Annotations.XY_SHIFT,
    Column.Annotations.Z_SHIFT,
    # Optical flow features ====================================================
    Column.OpticalFlow.UNIT_VECTOR_MEAN,
    Column.OpticalFlow.ANGLE_STD,
    Column.OpticalFlow.SPEED_MEAN,
    Column.OpticalFlow.SPEED_STD,
]
"""List of feature to include in TFE manifest."""
