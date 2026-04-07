"""Settings for working with Timelapse Feature Explorer (TFE)."""

from colorizer_data import FeatureInfo, FeatureType
from numpy import pi

from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE

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

TFE_REQUIRED_COLUMNS: list[str | Column.SegData | Column.SegDataFilters] = [
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

TFE_FEATURE_MAP = {
    # General information ======================================================
    Column.SegData.TIME_HRS: FeatureInfo(
        label="Time (hours)",
        type=FeatureType.CONTINUOUS,
        description="Time in hours",
    ),
    Column.SegData.TIME_MINS: FeatureInfo(
        label="Time (minutes)",
        type=FeatureType.CONTINUOUS,
        description="Time in minutes",
    ),
    Column.TRACK_ID: FeatureInfo(
        label="Track ID",
        type=FeatureType.DISCRETE,
    ),
    Column.TRACK_LENGTH: FeatureInfo(
        label="Track Duration",
        type=FeatureType.DISCRETE,
    ),
    # Classic segmentation features ============================================
    Column.SegData.ALIGNMENT: FeatureInfo(
        label="Alignment Relative to Flow (radians)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=pi / 2,
    ),
    Column.SegData.ALIGNMENT_DEG: FeatureInfo(
        label="Alignment Relative to Flow (degrees)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=90,
    ),
    Column.SegData.ORIENTATION: FeatureInfo(
        label="Cell Orientation (radians)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=pi,
    ),
    Column.SegData.ORIENTATION_DEG: FeatureInfo(
        label="Cell Orientation (degrees)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=180,
    ),
    Column.SegData.AREA_UM_SQ: FeatureInfo(
        label="Cell Area", type=FeatureType.CONTINUOUS, description="", unit="µm²"
    ),
    Column.SegData.ASPECT_RATIO: FeatureInfo(
        label="Aspect Ratio",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
    Column.SegData.PERIMETER_UM: FeatureInfo(
        label="Cell Perimeter", type=FeatureType.CONTINUOUS, description="", unit="µm"
    ),
    Column.SegData.ECCENTRICITY: FeatureInfo(
        label="Cell Eccentricity",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
    Column.SegData.CELL_FLUOR_MAX: FeatureInfo(
        label="Cell Fluorescence Max", type=FeatureType.CONTINUOUS, description="", unit="a.u."
    ),
    Column.SegData.CELL_FLUOR_MEAN: FeatureInfo(
        label="Cell Fluorescence Mean", type=FeatureType.CONTINUOUS, description="", unit="a.u."
    ),
    Column.SegData.CELL_FLUOR_MEDIAN: FeatureInfo(
        label="Cell Fluorescence Median", type=FeatureType.CONTINUOUS, description="", unit="a.u."
    ),
    Column.SegData.EDGE_FLUOR_MEAN: FeatureInfo(
        label="Edge Fluorescence Mean", type=FeatureType.CONTINUOUS, description="", unit="a.u."
    ),
    Column.SegData.NODE_FLUOR_MEAN: FeatureInfo(
        label="Node Fluorescence Mean", type=FeatureType.CONTINUOUS, description="", unit="a.u."
    ),
    Column.SegData.SOLIDITY: FeatureInfo(
        label="Cell Solidity",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
    Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF: FeatureInfo(
        label="Smoothed Area Difference (Normalized)",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
    Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK: FeatureInfo(
        label="Number of Valid Timepoints",
        type=FeatureType.DISCRETE,
        description="",
    ),
    Column.SegData.NUM_NEIGHBORS: FeatureInfo(
        label="Number of Neighbors",
        type=FeatureType.DISCRETE,
        description="",
    ),
    Column.SegData.NUM_NUCLEI_IN_CROP: FeatureInfo(
        label="Number of Nuclei in Crop",
        type=FeatureType.DISCRETE,
        description="",
    ),
    # Dynamic features =========================================================
    Column.SegData.CENTROID_VELOCITY_ANGLE_DEG: FeatureInfo(
        label="Cell Migration Angle", type=FeatureType.CONTINUOUS, description="", unit="degrees"
    ),
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN: FeatureInfo(
        label="Cell Migration Speed", type=FeatureType.CONTINUOUS, description="", unit="µm/min"
    ),
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN_SMOOTHED: FeatureInfo(
        label="Cell Migration Speed (Smoothed)",
        type=FeatureType.CONTINUOUS,
        description="",
        unit="µm/min",
    ),
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG: FeatureInfo(
        label="Nucleus Orientation Relative to Migration",
        type=FeatureType.CONTINUOUS,
        description="",
        unit="degrees",
    ),
    Column.SegData.NUCLEI_POSITION_ANGLE_DEG: FeatureInfo(
        label="Nucleus Orientation Relative to Flow Angle",
        type=FeatureType.CONTINUOUS,
        description="",
        unit="degrees",
    ),
    Column.SegData.VECTOR_MEAN_FOR_CROP_MAGNITUDE: FeatureInfo(
        label="Migration Coherence in Crop (Vector Mean Magnitude)",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
    # DiffAE-based features ====================================================
    **{
        f"{Column.DiffAEData.PCA_FEATURE_PREFIX}{i}": FeatureInfo(
            label=f"PC {i}",
            type=FeatureType.CONTINUOUS,
            description=f"Principal component {i} calculated from DiffAE model latent features",
        )
        for i in range(1, MAX_PCS_TO_COMPUTE + 1)
    },
    Column.DiffAEData.POLAR_ANGLE: FeatureInfo(
        label="PC Polar Angle",
        type=FeatureType.CONTINUOUS,
        description="Polar angle calculated by transforming PC 1 and PC 2 to polar coordinates",
        min=0,
        max=pi,
    ),
    Column.DiffAEData.POLAR_RADIUS: FeatureInfo(
        label="PC Polar Radius",
        type=FeatureType.CONTINUOUS,
        description="Polar radius calculated by transforming PC 1 and PC 2 to polar coordinates",
    ),
    Column.DiffAEData.PC3_FLIPPED: FeatureInfo(
        label="PC Rho",
        type=FeatureType.CONTINUOUS,
        description="Negative value of PC 3",
    ),
    # Segmentation filters =====================================================
    Column.SegDataFilters.IS_EDGE_SEGMENTATION: FeatureInfo(
        label="Filter: Touches Edge of Field of View",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="True if segmentation touches edge of FOV, False otherwise",
    ),
    Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE: FeatureInfo(
        label="Filter: Smoothed Area Change Below Threshold",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="True if smoothed area change is below threshold value, False otherwise",
    ),
    Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION: FeatureInfo(
        label="Filter: Exceeds Min Track Duration",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="True if track duration is greater than minimum duration, False otherwise",
    ),
    Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK: FeatureInfo(
        label="Filter: Num Valid Points Exceeds Threshold",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="True if track has more points than minimum threshold value, False otherwise",
    ),
    Column.SegDataFilters.IS_INCLUDED: FeatureInfo(
        label="Filter: Passed All Filters",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="True if segmentation passes all filters, False otherwise",
    ),
    Column.SegDataFilters.IS_VALID_BBOX: FeatureInfo(
        label="Annotation: Crop Box Limits are Within FOV",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="True if crop bounding box are within the FOV, False otherwise",
    ),
    # Timepoint annotations ====================================================
    Column.Annotations.AUTO_BF_SCOPE_ERROR: FeatureInfo(
        label="Filter: Auto-detected Brightfield Microscope Error",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Auto detected error with brightfield scope",
    ),
    Column.Annotations.AUTO_BF_TEMP_ARTIFACT: FeatureInfo(
        label="Filter: Auto-detected Temporary Artifact",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Auto detected temporary brightfield artifact.",
    ),
    Column.Annotations.AUTO_GFP_SCOPE_ERROR: FeatureInfo(
        label="Filter: Auto-detected GFP Channel Microscope Error",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Auto detected error with GFP scope",
    ),
    Column.Annotations.BF_SCOPE_ERROR: FeatureInfo(
        label="Filter: Manually Annotated Brightfield Microscope Error",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Manually annotated error with brightfield scope",
    ),
    Column.Annotations.BF_TEMP_ARTIFACT: FeatureInfo(
        label="Filter: Manually Annotated Temporary Artifact",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Manually annotated temporary brightfield artifact",
    ),
    Column.Annotations.GFP_SCOPE_ERROR: FeatureInfo(
        label="Filter: Manually Annotated GFP Channel Microscope Error",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Manually annotated error with GFP scope",
    ),
    Column.Annotations.CELL_PILING: FeatureInfo(
        label="Filter: Manually Annotated Significant Cell Piling",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Manually annotated range of timepoints where cells pile up (> 30% of FOV)",
    ),
    Column.Annotations.NOT_STEADY_STATE: FeatureInfo(
        label="Filter: Cells Not At Steady State",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Timepoint is not at visual steady state",
    ),
    Column.Annotations.UNFED: FeatureInfo(
        label="Filter: Unfed (More Than 3 Hours Since Fresh Media Introduced)",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Manually annotated timepoint where cells are more than 3hrs since last feeding",
    ),
    Column.Annotations.XY_SHIFT: FeatureInfo(
        label="Filter: Significant Change in XY position of FOV",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Manually annotated shift in the XY position",
    ),
    Column.Annotations.Z_SHIFT: FeatureInfo(
        label="Filter: Significant Change in Z position of FOV",
        type=FeatureType.CATEGORICAL,
        categories=["False", "True"],
        description="Manually annotated shift in the Z focus.",
    ),
    # Optical flow features ====================================================
    "optical_flow_mean_unit_vector_dt1": FeatureInfo(
        label="Coherent Migration (Optical Flow Mean Unit Vector)",
        description="",
        type=FeatureType.CONTINUOUS,
        min=0,
        max=1,
    ),
    "optical_flow_angle_std_dt1": FeatureInfo(
        label="Coherent Migration (Optical Flow Angle Std Dev)",
        description="",
        type=FeatureType.CONTINUOUS,
        min=0,
        max=4,
    ),
    "optical_flow_mean_speed_dt1": FeatureInfo(
        label="Optical Flow Mean Speed",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=8,
    ),
    "optical_flow_std_speed_dt1": FeatureInfo(
        label="Optical Flow Speed Std Dev",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=10,
    ),
    "optical_flow_mean_unit_vector_fast": FeatureInfo(
        label="Coherent Migration Fast (Optical flow unit vectors greater than 1 speed)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=1,
    ),
    "speed_above_1_count": FeatureInfo(
        label="N vectors with Speed Above 1",
        type=FeatureType.DISCRETE,
        description="",
    ),
    "ema005_optical_flow_mean_unit_vector_dt1": FeatureInfo(
        label="Coherent Migration (EMA 0.05, Optical Flow Mean Unit Vector)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=1,
    ),
    "ema005_optical_flow_mean_unit_vector_fast_dt1": FeatureInfo(
        label="Coherent Migration (EMA 0.05, Optical Flow Mean Unit Vector Fast)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=1,
    ),
    "ema01_optical_flow_mean_unit_vector_dt1": FeatureInfo(
        label="Coherent Migration (EMA 0.1, Optical Flow Mean Unit Vector)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=1,
    ),
    "ema01_optical_flow_mean_unit_vector_fast_dt1": FeatureInfo(
        label="Coherent Migration (EMA 0.1, Optical Flow Mean Unit Vector Fast)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=1,
    ),
    "ema02_optical_flow_mean_unit_vector_dt1": FeatureInfo(
        label="Coherent Migration (EMA 0.2, Optical Flow Mean Unit Vector)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=1,
    ),
    "ema02_optical_flow_mean_unit_vector_fast_dt1": FeatureInfo(
        label="Coherent Migration (EMA 0.2, Optical Flow Mean Unit Vector Fast)",
        type=FeatureType.CONTINUOUS,
        description="",
        min=0,
        max=1,
    ),
    "ema01_optical_flow_radial_coherence_dt1": FeatureInfo(
        label="Coherent Migration (EMA 0.1, Optical Flow Radial Coherence)",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
    "ema01_optical_flow_radial_coherence_weighted_dt1": FeatureInfo(
        label="Coherent Migration (EMA 0.1, Optical Flow Radial Coherence Weighted)",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
    "optical_flow_radial_coherence_dt1": FeatureInfo(
        label="Coherent Migration (Optical Flow Radial Coherence)",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
    "optical_flow_radial_coherence_weighted_dt1": FeatureInfo(
        label="Coherent Migration (Optical Flow Radial Coherence Weighted)",
        type=FeatureType.CONTINUOUS,
        description="",
    ),
}
"""Map of feature information for TFE"""
