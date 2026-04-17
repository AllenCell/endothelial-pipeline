"""Feature metadata structure and mapping to column names."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from numpy import pi

from endo_pipeline.io import slugify
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

MIN_VALUE = Literal["min"]
"""Use minimum value from data for feature limits."""

MAX_VALUE = Literal["max"]
"""Use maximum value from data for feature limits."""


class FeatureType(StrEnum):
    """Feature type."""

    CONTINUOUS = "continuous"
    """Feature has continuous values."""

    DISCRETE = "discrete"
    """Feature only has discrete values."""

    BOOLEAN = "boolean"
    """Feature is boolean."""


@dataclass
class FeatureMetadata:
    """Feature metadata."""

    name: str
    """Full feature name in title case."""

    label: str | None = None
    """Short feature label in title case. If not provided, set equal to name."""

    unit: str | None = None
    """Unit of the feature."""

    description: str | None = None
    """Description of the feature."""

    min: float | MIN_VALUE | None = None
    """Minimum value for feature."""

    max: float | MAX_VALUE | None = None
    """Maximum value for feature."""

    type: FeatureType = FeatureType.CONTINUOUS
    """Feature type."""

    bin_width: float | None = None
    """Width of bins."""

    ticks: range | None = None
    """Range for ticks."""

    slug: str = field(init=False)
    """Slug version of name."""

    name_with_unit: str = field(init=False)
    """Feature name with unit appended."""

    label_with_unit: str = field(init=False)
    """Feature label with unit appended."""

    limits: tuple[float | None | MIN_VALUE, float | None | MAX_VALUE] = field(init=False)
    """Minimum and maximum values of the feature as a tuple."""

    def __post_init__(self):
        """Post initialization steps for feature metadata."""

        # If label is not provided, set equal to the name.
        if self.label is None:
            self.label = self.name

        # Create versions of the name and label with unit.
        unit = f" ({self.unit})" if self.unit else ""
        self.name_with_unit = f"{self.name}{unit}"
        self.label_with_unit = f"{self.label}{unit}"

        # Create slug version of the name (useful for saving to files).
        self.slug = slugify(self.name_with_unit)

        # Set limits using minimum and maximum.
        self.limits = (self.min, self.max)


FEATURE_METADATA = {
    # General information ======================================================
    Column.SegData.TIME_HRS: FeatureMetadata(
        name="Time",
        unit="hr",
        description="Time in hours",
        min=0,
        max="max",
        bin_width=0.5,
        ticks=range(0, 49, 12),
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.TIME_MINS: FeatureMetadata(
        name="Time",
        unit="min",
        description="Time in minutes",
        min=0,
        max="max",
        bin_width=30,
        ticks=range(0, 2881, 720),
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.TIME_HRS_SINCE_FLOW: FeatureMetadata(
        name="Time Under Flow",
        unit="hr",
        min="min",
        max="max",
        bin_width=0.5,
        ticks=range(0, 49, 12),
        type=FeatureType.CONTINUOUS,
    ),
    Column.TRACK_ID: FeatureMetadata(
        name="Track ID",
        type=FeatureType.DISCRETE,
    ),
    Column.TRACK_LENGTH: FeatureMetadata(
        name="Track Duration",
        type=FeatureType.DISCRETE,
    ),
    # Classic segmentation features ============================================
    Column.SegData.ALIGNMENT: FeatureMetadata(
        name="Alignment Relative to Flow",
        label="Cell Alignment",
        unit="rad",
        min=0,
        max=pi / 2,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.ALIGNMENT_DEG: FeatureMetadata(
        name="Alignment Relative to Flow",
        label="Cell Alignment",
        unit="°",
        min=0,
        max=90,
        bin_width=1,
        ticks=range(0, 91, 15),
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.ORIENTATION: FeatureMetadata(
        name="Cell Orientation",
        label="Orientation",
        unit="rad",
        min=0,
        max=pi,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.ORIENTATION_DEG: FeatureMetadata(
        name="Cell Orientation",
        label="Orientation",
        unit="°",
        min=0,
        max=180,
        bin_width=5,
        ticks=range(0, 181, 90),
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.NEMATIC_ORDER: FeatureMetadata(
        name="Nematic Order",
        min=-1,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.AREA_UM_SQ: FeatureMetadata(
        name="Cell Area",
        label="Area",
        unit=f"{Unicode.MU}m{Unicode.SQUARED}",
        min=350,
        max=2000,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.ASPECT_RATIO: FeatureMetadata(
        name="Aspect Ratio",
        min=1,
        max=10,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.PERIMETER_UM: FeatureMetadata(
        name="Cell Perimeter",
        label="Perimeter",
        unit=f"{Unicode.MU}m",
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.ECCENTRICITY: FeatureMetadata(
        name="Cell Eccentricity",
        label="Eccentricity",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MAX: FeatureMetadata(
        name="Max VE-Cad Fluorescence in Cell",
        label="Cell Fluorescence Max",
        unit="a.u.",
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MEAN: FeatureMetadata(
        name="Mean VE-Cad Fluorescence in Cell",
        label="Cell Fluorescence Mean",
        unit="a.u.",
        min=120,
        max=150,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MEDIAN: FeatureMetadata(
        name="Median VE-Cad Fluorescence in Cell",
        label="Cell Fluorescence Median",
        unit="a.u.",
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.EDGE_FLUOR_MEAN: FeatureMetadata(
        name="Mean VE-Cad Fluorescence at Edges",
        label="Edge Fluorescence Mean",
        unit="a.u.",
        min=100,
        max=200,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.NODE_FLUOR_MEAN: FeatureMetadata(
        name="Mean VE-Cad Fluorescence at Nodes",
        label="Node Fluorescence Mean",
        unit="a.u.",
        min=100,
        max=200,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.SOLIDITY: FeatureMetadata(
        name="Cell Solidity",
        label="Solidity",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF: FeatureMetadata(
        name="Smoothed Area Difference (Normalized)",
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK: FeatureMetadata(
        name="Number of Valid Timepoints",
        type=FeatureType.DISCRETE,
    ),
    Column.SegData.NUM_NEIGHBORS: FeatureMetadata(
        name="Number of Neighbors",
        label="Number of\nNeighbors",
        min=0,
        max="max",
        bin_width=1,
        type=FeatureType.DISCRETE,
    ),
    Column.SegData.NUCLEI_POSITION_DISTANCE: FeatureMetadata(
        name="Nuclei-Cell Centroid Distance",
        unit="px",
        min=0,
        max="max",
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.NUM_NUCLEI_AT_TIMEPOINT: FeatureMetadata(
        name="Number of Nuclei",
        min=0,
        type=FeatureType.DISCRETE,
    ),
    Column.SegData.NUM_NUCLEI_IN_CROP: FeatureMetadata(
        name="Number of Nuclei in Crop",
        label="Number of Nuclei\nin Crop",
        min=0,
        type=FeatureType.DISCRETE,
    ),
    # Dynamic features =========================================================
    Column.SegData.CENTROID_VELOCITY_ANGLE_DEG: FeatureMetadata(
        name="Cell Migration Angle",
        label="Migration Angle",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN: FeatureMetadata(
        name="Cell Migration Speed",
        label="Centroid Velocity\nMagnitude",
        unit=f"{Unicode.MU}m/min",
        min=0,
        max="max",
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN_SMOOTHED: FeatureMetadata(
        name="Cell Migration Speed (Smoothed)",
        unit=f"{Unicode.MU}m/min",
        min=0,
        max="max",
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG: FeatureMetadata(
        name="Nucleus Orientation Relative to Migration",
        label="Cell-Nucleus Angle\nRel. Migration",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD: FeatureMetadata(
        name="Cell-Nucleus vs. Migration Dot Product",
        label="Cell-Nucleus vs.\nMigration Dot Prod.",
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_ANGLE_DEG: FeatureMetadata(
        name="Nucleus Orientation Relative to Flow Angle",
        label="Cell-Nucleus Angle\nRel. to Flow",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=FeatureType.CONTINUOUS,
    ),
    Column.SegData.VECTOR_MEAN_FOR_CROP_MAGNITUDE: FeatureMetadata(
        name="Migration Coherence in Crop (Vector Mean Magnitude)",
        type=FeatureType.CONTINUOUS,
    ),
    # DiffAE-based features ====================================================
    **{
        f"{Column.DiffAEData.PCA_FEATURE_PREFIX}{i}": FeatureMetadata(
            name=f"PC {i}",
            description=f"Principal component {i} calculated from DiffAE model latent features",
            type=FeatureType.CONTINUOUS,
        )
        for i in range(1, MAX_PCS_TO_COMPUTE + 1)
    },
    Column.DiffAEData.POLAR_ANGLE: FeatureMetadata(
        name="PC Polar Angle",
        label=Unicode.THETA,
        description="Polar angle calculated by transforming PC 1 and PC 2 to polar coordinates",
        min=0,
        max=pi,
        type=FeatureType.CONTINUOUS,
    ),
    Column.DiffAEData.POLAR_RADIUS: FeatureMetadata(
        name="PC Polar Radius",
        label="r",
        description="Polar radius calculated by transforming PC 1 and PC 2 to polar coordinates",
        min=0,
        type=FeatureType.CONTINUOUS,
    ),
    Column.DiffAEData.PC3_FLIPPED: FeatureMetadata(
        name="PC Rho",
        label=Unicode.RHO,
        description="Negative value of PC 3",
        type=FeatureType.CONTINUOUS,
    ),
    # Segmentation filters =====================================================
    Column.SegDataFilters.IS_EDGE_SEGMENTATION: FeatureMetadata(
        name="Filter: Touches Edge of Field of View",
        description="True if segmentation touches edge of FOV, False otherwise",
        type=FeatureType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE: FeatureMetadata(
        name="Filter: Smoothed Area Change Below Threshold",
        description="True if smoothed area change is below threshold value, False otherwise",
        type=FeatureType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION: FeatureMetadata(
        name="Filter: Exceeds Min Track Duration",
        description="True if track duration is greater than minimum duration, False otherwise",
        type=FeatureType.BOOLEAN,
    ),
    Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK: FeatureMetadata(
        name="Filter: Num Valid Points Exceeds Threshold",
        description="True if track has more points than minimum threshold value, False otherwise",
        type=FeatureType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_INCLUDED: FeatureMetadata(
        name="Filter: Passed All Filters",
        description="True if segmentation passes all filters, False otherwise",
        type=FeatureType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_VALID_BBOX: FeatureMetadata(
        name="Filter: Crop Box Limits are Within FOV",
        description="True if crop bounding box are within the FOV, False otherwise",
        type=FeatureType.BOOLEAN,
    ),
    Column.SegData.NUM_TRACKS_BEFORE_FILTERING: FeatureMetadata(
        name="Number of Segmentations Before Filtering",
        label="Num. Segmentations\nBefore Filtering",
        min=0,
        type=FeatureType.DISCRETE,
    ),
    Column.SegData.NUM_TRACKS_AFTER_FILTERING: FeatureMetadata(
        name="Number of Segmentations After Filtering",
        label="Num. Segmentations\nAfter Filtering",
        min=0,
        type=FeatureType.DISCRETE,
    ),
    # Timepoint annotations ====================================================
    Column.Annotations.AUTO_BF_SCOPE_ERROR: FeatureMetadata(
        name="Filter: Auto-detected Brightfield Microscope Error",
        description="Auto detected error with brightfield scope",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.AUTO_BF_TEMP_ARTIFACT: FeatureMetadata(
        name="Filter: Auto-detected Temporary Artifact",
        description="Auto detected temporary brightfield artifact.",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.AUTO_GFP_SCOPE_ERROR: FeatureMetadata(
        name="Filter: Auto-detected GFP Channel Microscope Error",
        description="Auto detected error with GFP scope",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.BF_SCOPE_ERROR: FeatureMetadata(
        name="Filter: Manually Annotated Brightfield Microscope Error",
        description="Manually annotated error with brightfield scope",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.BF_TEMP_ARTIFACT: FeatureMetadata(
        name="Filter: Manually Annotated Temporary Artifact",
        description="Manually annotated temporary brightfield artifact",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.GFP_SCOPE_ERROR: FeatureMetadata(
        name="Filter: Manually Annotated GFP Channel Microscope Error",
        description="Manually annotated error with GFP scope",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.CELL_PILING: FeatureMetadata(
        name="Filter: Manually Annotated Significant Cell Piling",
        description="Manually annotated range of timepoints where cells pile up (> 30% of FOV)",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.NOT_STEADY_STATE: FeatureMetadata(
        name="Filter: Cells Not At Steady State",
        description="Timepoint is not at visual steady state",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.UNFED: FeatureMetadata(
        name="Filter: Unfed (More Than 3 Hours Since Fresh Media Introduced)",
        description="Manually annotated timepoint where cells are more than 3hrs since last feeding",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.XY_SHIFT: FeatureMetadata(
        name="Filter: Significant Change in XY position of FOV",
        description="Manually annotated shift in the XY position",
        type=FeatureType.BOOLEAN,
    ),
    Column.Annotations.Z_SHIFT: FeatureMetadata(
        name="Filter: Significant Change in Z position of FOV",
        description="Manually annotated shift in the Z focus.",
        type=FeatureType.BOOLEAN,
    ),
    # Optical flow features ====================================================
    "optical_flow_mean_unit_vector_dt1": FeatureMetadata(
        name="Coherent Migration (Optical Flow Mean Unit Vector)",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    Column.OpticalFlow.ANGLE_MEAN: FeatureMetadata(
        name="Optical Flow Mean Angle",
        min=0,
        max=8,
        type=FeatureType.CONTINUOUS,
    ),
    Column.OpticalFlow.ANGLE_STD: FeatureMetadata(
        name="Coherent Migration (Optical Flow Angle Std Dev)",
        min=0,
        max=4,
        type=FeatureType.CONTINUOUS,
    ),
    Column.OpticalFlow.SPEED_MEAN: FeatureMetadata(
        name="Optical Flow Mean Speed",
        label="Mean Speed",
        min=0,
        max=8,
        type=FeatureType.CONTINUOUS,
    ),
    Column.OpticalFlow.SPEED_STD: FeatureMetadata(
        name="Optical Flow Speed Std Dev",
        min=0,
        max=10,
        type=FeatureType.CONTINUOUS,
    ),
    "optical_flow_mean_unit_vector_fast": FeatureMetadata(
        name="Coherent Migration Fast (Optical flow unit vectors greater than 1 speed)",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    "speed_above_1_count": FeatureMetadata(
        name="N vectors with Speed Above 1",
        type=FeatureType.DISCRETE,
    ),
    "ema005_optical_flow_mean_unit_vector_dt1": FeatureMetadata(
        name="Coherent Migration (EMA 0.05, Optical Flow Mean Unit Vector)",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    "ema005_optical_flow_mean_unit_vector_fast_dt1": FeatureMetadata(
        name="Coherent Migration (EMA 0.05, Optical Flow Mean Unit Vector Fast)",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    Column.OpticalFlow.UNIT_VECTOR_MEAN: FeatureMetadata(
        name="Coherent Migration (EMA 0.1, Optical Flow Mean Unit Vector)",
        label="Migration Coherence",
        min=0,
        max=1,
        bin_width=0.02,
        type=FeatureType.CONTINUOUS,
    ),
    "ema01_optical_flow_mean_unit_vector_fast_dt1": FeatureMetadata(
        name="Coherent Migration (EMA 0.1, Optical Flow Mean Unit Vector Fast)",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    "ema02_optical_flow_mean_unit_vector_dt1": FeatureMetadata(
        name="Coherent Migration (EMA 0.2, Optical Flow Mean Unit Vector)",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    "ema02_optical_flow_mean_unit_vector_fast_dt1": FeatureMetadata(
        name="Coherent Migration (EMA 0.2, Optical Flow Mean Unit Vector Fast)",
        min=0,
        max=1,
        type=FeatureType.CONTINUOUS,
    ),
    "ema01_optical_flow_radial_coherence_dt1": FeatureMetadata(
        name="Coherent Migration (EMA 0.1, Optical Flow Radial Coherence)",
        type=FeatureType.CONTINUOUS,
    ),
    "ema01_optical_flow_radial_coherence_weighted_dt1": FeatureMetadata(
        name="Coherent Migration (EMA 0.1, Optical Flow Radial Coherence Weighted)",
        type=FeatureType.CONTINUOUS,
    ),
    "optical_flow_radial_coherence_dt1": FeatureMetadata(
        name="Coherent Migration (Optical Flow Radial Coherence)",
        type=FeatureType.CONTINUOUS,
    ),
    "optical_flow_radial_coherence_weighted_dt1": FeatureMetadata(
        name="Coherent Migration (Optical Flow Radial Coherence Weighted)",
        type=FeatureType.CONTINUOUS,
    ),
}
"""Mapping of column names to feature metadata."""


FEATURE_METADATA_DICT: dict[str, dict[str, Any]] = {
    Column.SegData.TIME_HRS: {
        "column_name": Column.SegData.TIME_HRS,
        "label": "Time (h)",
        "lims": (0, "max"),
        "bin_width": 0.5,
        "ticks": range(0, 49, 12),
        "discrete_ticks": False,
    },
    Column.SegData.TIME_HRS_SINCE_FLOW: {
        "column_name": Column.SegData.TIME_HRS_SINCE_FLOW,
        "label": "Time Under Flow (h)",
        "lims": ("min", "max"),
        "bin_width": 0.5,
        "ticks": range(0, 49, 12),
        "discrete_ticks": False,
    },
    Column.SegData.ALIGNMENT_DEG: {
        "column_name": Column.SegData.ALIGNMENT_DEG,
        "label": "Alignment (deg)",
        "lims": (0, 90),
        "bin_width": 1,
        "ticks": range(0, 91, 15),
        "discrete_ticks": False,
    },
    Column.SegData.ORIENTATION_DEG: {
        "column_name": Column.SegData.ORIENTATION_DEG,
        "label": "Orientation (deg)",
        "lims": (0, 180),
        "bin_width": 5,
        "ticks": range(0, 181, 90),
        "discrete_ticks": False,
    },
    Column.SegData.ORIENTATION: {
        "column_name": Column.SegData.ORIENTATION,
        "label": "Orientation",
        "lims": (0, 180),
        "bin_width": 5,
        "ticks": range(0, 181, 90),
        "discrete_ticks": False,
    },
    Column.SegData.NEMATIC_ORDER: {
        "column_name": Column.SegData.NEMATIC_ORDER,
        "label": "Nematic Order",
        "lims": (-1, 1),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.ECCENTRICITY: {
        "column_name": Column.SegData.ECCENTRICITY,
        "label": "Eccentricity",
        "lims": (0, 1),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.ASPECT_RATIO: {
        "column_name": Column.SegData.ASPECT_RATIO,
        "label": "Aspect Ratio",
        "lims": (1, 10),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.AREA_UM_SQ: {
        "column_name": Column.SegData.AREA_UM_SQ,
        "label": "Area",
        "lims": (350, 2000),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.NUM_NEIGHBORS: {
        "column_name": Column.SegData.NUM_NEIGHBORS,
        "label": "Number of\nNeighbors",
        "lims": (0, "max"),
        "bin_width": 1,
        "ticks": None,
        "discrete_ticks": True,
    },
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN: {
        "column_name": Column.SegData.CENTROID_VELOCITY_UM_PER_MIN,
        "label": "Centroid Velocity\nMagnitude (μm/min)",
        "lims": (0, "max"),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.CENTROID_VELOCITY_ANGLE_DEG: {
        "column_name": Column.SegData.CENTROID_VELOCITY_ANGLE_DEG,
        "label": "Centroid Velocity\nOrientation (deg)",
        "lims": (-180, 181),
        "bin_width": 5,
        "ticks": range(-180, 181, 90),
        "discrete_ticks": False,
    },
    Column.SegData.NUCLEI_POSITION_ANGLE_DEG: {
        "column_name": Column.SegData.NUCLEI_POSITION_ANGLE_DEG,
        "label": "Nuclei Orientation\nRel. to Flow (deg)",
        "lims": (-180, 180),
        "bin_width": 5,
        "ticks": range(-180, 181, 90),
        "discrete_ticks": False,
    },
    Column.SegData.NUCLEI_POSITION_DISTANCE: {
        "column_name": Column.SegData.NUCLEI_POSITION_DISTANCE,
        "label": "Nuclei-Cell Centroid Distance (px)",
        "lims": (0, "max"),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.NUM_NUCLEI_AT_TIMEPOINT: {
        "column_name": Column.SegData.NUM_NUCLEI_AT_TIMEPOINT,
        "label": "Number of Nuclei",
        "lims": (0, None),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": True,
    },
    Column.SegData.NUM_NUCLEI_IN_CROP: {
        "column_name": Column.SegData.NUM_NUCLEI_IN_CROP,
        "label": "Number of Nuclei\nin Crop",
        "lims": (0, None),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": True,
    },
    Column.SegData.CELL_FLUOR_MEAN: {
        "column_name": Column.SegData.CELL_FLUOR_MEAN,
        "label": "Mean Cell Fluorescence",
        "lims": (120, 150),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.EDGE_FLUOR_MEAN: {
        "column_name": Column.SegData.EDGE_FLUOR_MEAN,
        "label": "Mean Edge Fluorescence",
        "lims": (100, 200),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.NODE_FLUOR_MEAN: {
        "column_name": Column.SegData.NODE_FLUOR_MEAN,
        "label": "Mean Node Fluorescence",
        "lims": (100, 200),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.SOLIDITY: {
        "column_name": Column.SegData.SOLIDITY,
        "label": "Cell Solidity",
        "lims": (0, 1),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG: {
        "column_name": Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG,
        "label": "Nuclei Orientation\nRel. to Migration (deg)",
        "lims": (-180, 180),
        "bin_width": 5,
        "ticks": range(-180, 181, 90),
        "discrete_ticks": False,
    },
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD: {
        "column_name": Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD,
        "label": "Cell-Nucleus vs.\nMigration Dot Product",
        "lims": (None, None),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.DiffAEData.POLAR_RADIUS: {
        "column_name": Column.DiffAEData.POLAR_RADIUS,
        "label": "r",
        "lims": (0, None),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.DiffAEData.POLAR_ANGLE: {
        "column_name": Column.DiffAEData.POLAR_ANGLE,
        "label": Unicode.THETA,
        "lims": None,
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.DiffAEData.PC3_FLIPPED: {
        "column_name": Column.DiffAEData.PC3_FLIPPED,
        "label": Unicode.RHO,
        "lims": None,
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.SegData.NUM_TRACKS_BEFORE_FILTERING: {
        "column_name": Column.SegData.NUM_TRACKS_BEFORE_FILTERING,
        "label": "Num. Segmentations\nBefore Filtering",
        "lims": (0, None),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": True,
    },
    Column.SegData.NUM_TRACKS_AFTER_FILTERING: {
        "column_name": Column.SegData.NUM_TRACKS_AFTER_FILTERING,
        "label": "Num. Segmentations\nAfter Filtering",
        "lims": (0, None),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": True,
    },
    Column.OpticalFlow.UNIT_VECTOR_MEAN: {
        "column_name": Column.OpticalFlow.UNIT_VECTOR_MEAN,
        "label": "Migration Coherence",
        "lims": (0, 1),
        "bin_width": 0.02,
        "ticks": None,
        "discrete_ticks": False,
    },
    Column.OpticalFlow.SPEED_MEAN: {
        "column_name": Column.OpticalFlow.SPEED_MEAN,
        "label": "Mean Speed",
        "lims": (0, None),
        "bin_width": None,
        "ticks": None,
        "discrete_ticks": False,
    },
}
