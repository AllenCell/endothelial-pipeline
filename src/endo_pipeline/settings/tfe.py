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

TFE_FEATURE_MAP = {
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
}
"""Map of feature information for TFE"""
