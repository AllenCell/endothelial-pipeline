"""Column metadata structure and mapping to column names."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from numpy import pi

from endo_pipeline.io import slugify
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE, NUM_LATENT_FEATURES
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

MIN_VALUE = Literal["min"]
"""Use minimum value from data for column limits."""

MAX_VALUE = Literal["max"]
"""Use maximum value from data for column limits."""


class ColumnType(StrEnum):
    """Column data type."""

    CONTINUOUS = "continuous"
    """Column has continuous values."""

    DISCRETE = "discrete"
    """Column only has discrete values."""

    BOOLEAN = "boolean"
    """Column is boolean."""


@dataclass
class ColumnMetadata:
    """Column metadata."""

    name: str
    """Full column name in title case."""

    label: str | None = None
    """Short column label in title case. If not provided, set equal to name."""

    unit: str | None = None
    """Unit of the column."""

    description: str | None = None
    """Description of the column."""

    min: float | MIN_VALUE | None = None
    """Minimum value for column."""

    max: float | MAX_VALUE | None = None
    """Maximum value for column."""

    type: ColumnType = ColumnType.CONTINUOUS
    """Column type."""

    bin_width: float | None = None
    """Width of bins."""

    ticks: range | None = None
    """Range for ticks."""

    slug: str = field(init=False)
    """Slug version of name."""

    name_with_unit: str = field(init=False)
    """Column name with unit appended."""

    label_with_unit: str = field(init=False)
    """Column label with unit appended."""

    limits: tuple[float | None | MIN_VALUE, float | None | MAX_VALUE] = field(init=False)
    """Minimum and maximum values of the column as a tuple."""

    def __post_init__(self):
        """Post initialization steps for column metadata."""

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


COLUMN_METADATA: dict[str, ColumnMetadata] = {
    # General information ======================================================
    Column.SegData.TIME_HRS: ColumnMetadata(
        name="Time",
        unit="hr",
        description="Time in hours",
        min=0,
        max="max",
        bin_width=0.5,
        ticks=range(0, 49, 12),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.TIME_MINS: ColumnMetadata(
        name="Time",
        unit="min",
        description="Time in minutes",
        min=0,
        max="max",
        bin_width=30,
        ticks=range(0, 2881, 720),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.TIME_HRS_SINCE_FLOW: ColumnMetadata(
        name="Time Under Flow",
        unit="hr",
        min="min",
        max="max",
        bin_width=0.5,
        ticks=range(0, 49, 12),
        type=ColumnType.CONTINUOUS,
    ),
    Column.TRACK_ID: ColumnMetadata(name="Track ID", type=ColumnType.DISCRETE),
    Column.TRACK_LENGTH: ColumnMetadata(name="Track Duration", type=ColumnType.DISCRETE),
    Column.DATASET: ColumnMetadata(name="Dataset", type=ColumnType.DISCRETE),
    Column.POSITION: ColumnMetadata(name="Position", type=ColumnType.DISCRETE),
    Column.TIMEPOINT: ColumnMetadata(name="Frame Number", type=ColumnType.DISCRETE),
    Column.CROP_INDEX: ColumnMetadata(name="Crop Index", type=ColumnType.DISCRETE),
    Column.SegData.LABEL: ColumnMetadata(name="Segmentation Label", type=ColumnType.DISCRETE),
    # Classic segmentation features ============================================
    Column.SegData.ALIGNMENT: ColumnMetadata(
        name="Alignment Relative to Flow",
        label="Cell Alignment Rel. to Flow",
        unit="rad",
        min=0,
        max=pi / 2,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ALIGNMENT_DEG: ColumnMetadata(
        name="Alignment Relative to Flow",
        label="Cell Alignment Rel. to Flow",
        unit="°",
        min=0,
        max=90,
        bin_width=1,
        ticks=range(0, 91, 15),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ORIENTATION: ColumnMetadata(
        name="Cell Orientation",
        label="Orientation",
        unit="rad",
        min=0,
        max=pi,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ORIENTATION_DEG: ColumnMetadata(
        name="Cell Orientation",
        label="Cell Orientation",
        unit="°",
        min=0,
        max=180,
        bin_width=5,
        ticks=range(0, 181, 90),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NEMATIC_ORDER: ColumnMetadata(
        name="Nematic Order",
        min=-1,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.AREA_UM_SQ: ColumnMetadata(
        name="Cell Area",
        label="Area",
        unit=f"{Unicode.MU}m{Unicode.SQUARED}",
        min=350,
        max=2000,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ASPECT_RATIO: ColumnMetadata(
        name="Cell Aspect Ratio",
        min=1,
        max=10,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.PERIMETER_UM: ColumnMetadata(
        name="Cell Perimeter",
        label="Perimeter",
        unit=f"{Unicode.MU}m",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ECCENTRICITY: ColumnMetadata(
        name="Cell Eccentricity",
        label="Eccentricity",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MAX: ColumnMetadata(
        name="Max VE-Cad Fluorescence in Cell",
        label="Cell Fluorescence Max",
        unit="a.u.",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MEAN: ColumnMetadata(
        name="Mean VE-Cad Fluorescence in Cell",
        label="Mean Cell Fluorescence",
        unit="a.u.",
        min=120,
        max=150,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MEDIAN: ColumnMetadata(
        name="Median VE-Cad Fluorescence in Cell",
        label="Cell Fluorescence Median",
        unit="a.u.",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.EDGE_FLUOR_MEAN: ColumnMetadata(
        name="Mean VE-Cad Fluorescence at Edges",
        label="Mean Edge Fluorescence",
        unit="a.u.",
        min=100,
        max=200,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NODE_FLUOR_MEAN: ColumnMetadata(
        name="Mean VE-Cad Fluorescence at Nodes",
        label="Node Fluorescence Mean",
        unit="a.u.",
        min=100,
        max=200,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.SOLIDITY: ColumnMetadata(
        name="Cell Solidity",
        label="Solidity",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF: ColumnMetadata(
        name="Smoothed Area Difference (Normalized)",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK: ColumnMetadata(
        name="Number of Valid Timepoints",
        type=ColumnType.DISCRETE,
    ),
    Column.SegData.NUM_NEIGHBORS: ColumnMetadata(
        name="Number of Neighbors",
        label="Number of\nNeighbors",
        min=0,
        max="max",
        bin_width=1,
        type=ColumnType.DISCRETE,
    ),
    Column.SegData.NUCLEI_POSITION_DISTANCE: ColumnMetadata(
        name="Nuclei-Cell Centroid Distance",
        unit="px",
        min=0,
        max="max",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NUM_NUCLEI_AT_TIMEPOINT: ColumnMetadata(
        name="Number of Nuclei",
        min=0,
        type=ColumnType.DISCRETE,
    ),
    Column.SegData.NUM_NUCLEI_IN_CROP: ColumnMetadata(
        name="Number of Nuclei in Crop",
        label="Number of Nuclei in Patch",
        min=0,
        type=ColumnType.DISCRETE,
    ),
    # Dynamic features =========================================================
    Column.SegData.CENTROID_VELOCITY_ANGLE_DEG: ColumnMetadata(
        name="Cell Migration Angle",
        label=f"Cell Migration {Unicode.THETA}",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN: ColumnMetadata(
        name="Cell Migration Speed",
        label="Cell Speed",
        unit=f"{Unicode.MU}m/min",
        min=0,
        max="max",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN_SMOOTHED: ColumnMetadata(
        name="Cell Migration Speed (Smoothed)",
        unit=f"{Unicode.MU}m/min",
        min=0,
        max="max",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG: ColumnMetadata(
        name="Nucleus Orientation Relative to Migration",
        label=f"Cell-Nuc {Unicode.THETA} Rel. Migration",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD: ColumnMetadata(
        name="Cell-Nucleus vs. Migration Dot Product",
        label="Cell-Nuc vs.\nMigration Dot Prod.",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_ANGLE_DEG: ColumnMetadata(
        name="Nucleus Orientation Relative to Flow Angle",
        label=f"Cell-Nuc {Unicode.THETA} Rel. to Flow",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.VECTOR_MEAN_FOR_CROP_MAGNITUDE: ColumnMetadata(
        name="Migration Coherence in Crop (Vector Mean Magnitude)",
        type=ColumnType.CONTINUOUS,
    ),
    # DiffAE-based features ====================================================
    **{
        f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}": ColumnMetadata(
            name=f"Feature {i}",
            description=f"DiffAE model latent feature {i}",
            type=ColumnType.CONTINUOUS,
        )
        for i in range(NUM_LATENT_FEATURES)
    },
    **{
        f"{Column.DiffAEData.PCA_FEATURE_PREFIX}{i}": ColumnMetadata(
            name=f"PC {i}",
            description=f"Principal component {i} calculated from DiffAE model latent features",
            type=ColumnType.CONTINUOUS,
        )
        for i in range(1, MAX_PCS_TO_COMPUTE + 1)
    },
    Column.DiffAEData.POLAR_ANGLE: ColumnMetadata(
        name="PC Polar Angle",
        label=Unicode.THETA,
        description="Polar angle calculated by transforming PC 1 and PC 2 to polar coordinates",
        min=0,
        max=pi,
        type=ColumnType.CONTINUOUS,
    ),
    Column.DiffAEData.POLAR_RADIUS: ColumnMetadata(
        name="PC Polar Radius",
        label="r",
        description="Polar radius calculated by transforming PC 1 and PC 2 to polar coordinates",
        min=0,
        type=ColumnType.CONTINUOUS,
    ),
    Column.DiffAEData.PC3_FLIPPED: ColumnMetadata(
        name="PC Rho",
        label=Unicode.RHO,
        description="Negative value of PC 3",
        type=ColumnType.CONTINUOUS,
    ),
    # Segmentation filters =====================================================
    Column.SegDataFilters.IS_EDGE_SEGMENTATION: ColumnMetadata(
        name="Filter: Touches Edge of Field of View",
        description="True if segmentation touches edge of FOV, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE: ColumnMetadata(
        name="Filter: Smoothed Area Change Below Threshold",
        description="True if smoothed area change is below threshold value, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION: ColumnMetadata(
        name="Filter: Exceeds Min Track Duration",
        description="True if track duration is greater than minimum duration, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK: ColumnMetadata(
        name="Filter: Num Valid Points Exceeds Threshold",
        description="True if track has more points than minimum threshold value, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_INCLUDED: ColumnMetadata(
        name="Filter: Passed All Filters",
        description="True if segmentation passes all filters, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_VALID_BBOX: ColumnMetadata(
        name="Filter: Crop Box Limits are Within FOV",
        description="True if crop bounding box are within the FOV, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegData.NUM_TRACKS_BEFORE_FILTERING: ColumnMetadata(
        name="Number of Segmentations Before Filtering",
        label="Num. Segmentations\nBefore Filtering",
        min=0,
        type=ColumnType.DISCRETE,
    ),
    Column.SegData.NUM_TRACKS_AFTER_FILTERING: ColumnMetadata(
        name="Number of Segmentations After Filtering",
        label="Num. Segmentations\nAfter Filtering",
        min=0,
        type=ColumnType.DISCRETE,
    ),
    # Timepoint annotations ====================================================
    Column.Annotations.AUTO_BF_SCOPE_ERROR: ColumnMetadata(
        name="Filter: Auto-detected Brightfield Microscope Error",
        description="Auto detected error with brightfield scope",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.AUTO_BF_TEMP_ARTIFACT: ColumnMetadata(
        name="Filter: Auto-detected Temporary Artifact",
        description="Auto detected temporary brightfield artifact.",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.AUTO_GFP_SCOPE_ERROR: ColumnMetadata(
        name="Filter: Auto-detected GFP Channel Microscope Error",
        description="Auto detected error with GFP scope",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.BF_SCOPE_ERROR: ColumnMetadata(
        name="Filter: Manually Annotated Brightfield Microscope Error",
        description="Manually annotated error with brightfield scope",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.BF_TEMP_ARTIFACT: ColumnMetadata(
        name="Filter: Manually Annotated Temporary Artifact",
        description="Manually annotated temporary brightfield artifact",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.GFP_SCOPE_ERROR: ColumnMetadata(
        name="Filter: Manually Annotated GFP Channel Microscope Error",
        description="Manually annotated error with GFP scope",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.CELL_PILING: ColumnMetadata(
        name="Filter: Manually Annotated Significant Cell Piling",
        description="Manually annotated range of timepoints where cells pile up (> 30% of FOV)",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.NOT_STEADY_STATE: ColumnMetadata(
        name="Filter: Cells Not At Steady State",
        description="Timepoint is not at visual steady state",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.UNFED: ColumnMetadata(
        name="Filter: Unfed (More Than 3 Hours Since Fresh Media Introduced)",
        description="Manually annotated timepoint where cells are more than 3hrs since last feeding",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.XY_SHIFT: ColumnMetadata(
        name="Filter: Significant Change in XY position of FOV",
        description="Manually annotated shift in the XY position",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.Z_SHIFT: ColumnMetadata(
        name="Filter: Significant Change in Z position of FOV",
        description="Manually annotated shift in the Z focus.",
        type=ColumnType.BOOLEAN,
    ),
    # Optical flow features ====================================================
    "optical_flow_mean_unit_vector_dt1": ColumnMetadata(
        name="Coherent Migration (Optical Flow Mean Unit Vector)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.ANGLE_MEAN: ColumnMetadata(
        name="Optical Flow Mean Angle",
        unit="rad",
        min=0,
        max=8,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.ANGLE_STD: ColumnMetadata(
        name="Coherent Migration (Optical Flow Angle Std Dev)",
        min=0,
        max=4,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.SPEED_MEAN: ColumnMetadata(
        name="Optical Flow Mean Speed",
        label="Mean Speed",
        unit="pixels/frame",
        min=0,
        max=8,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.SPEED_STD: ColumnMetadata(
        name="Optical Flow Speed Std Dev",
        min=0,
        max=10,
        type=ColumnType.CONTINUOUS,
    ),
    "optical_flow_mean_unit_vector_fast": ColumnMetadata(
        name="Coherent Migration Fast (Optical flow unit vectors greater than 1 speed)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "speed_above_1_count": ColumnMetadata(
        name="N vectors with Speed Above 1",
        type=ColumnType.DISCRETE,
    ),
    "ema005_optical_flow_mean_unit_vector_dt1": ColumnMetadata(
        name="Coherent Migration (EMA 0.05, Optical Flow Mean Unit Vector)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "ema005_optical_flow_mean_unit_vector_fast_dt1": ColumnMetadata(
        name="Coherent Migration (EMA 0.05, Optical Flow Mean Unit Vector Fast)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.UNIT_VECTOR_MEAN: ColumnMetadata(
        name="Coherent Migration (EMA 0.1, Optical Flow Mean Unit Vector)",
        label="Migration Coherence",
        min=0,
        max=1,
        bin_width=0.02,
        type=ColumnType.CONTINUOUS,
    ),
    "ema01_optical_flow_mean_unit_vector_fast_dt1": ColumnMetadata(
        name="Coherent Migration (EMA 0.1, Optical Flow Mean Unit Vector Fast)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "ema02_optical_flow_mean_unit_vector_dt1": ColumnMetadata(
        name="Coherent Migration (EMA 0.2, Optical Flow Mean Unit Vector)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "ema02_optical_flow_mean_unit_vector_fast_dt1": ColumnMetadata(
        name="Coherent Migration (EMA 0.2, Optical Flow Mean Unit Vector Fast)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "ema01_optical_flow_radial_coherence_dt1": ColumnMetadata(
        name="Coherent Migration (EMA 0.1, Optical Flow Radial Coherence)",
        type=ColumnType.CONTINUOUS,
    ),
    "ema01_optical_flow_radial_coherence_weighted_dt1": ColumnMetadata(
        name="Coherent Migration (EMA 0.1, Optical Flow Radial Coherence Weighted)",
        type=ColumnType.CONTINUOUS,
    ),
    "optical_flow_radial_coherence_dt1": ColumnMetadata(
        name="Coherent Migration (Optical Flow Radial Coherence)",
        type=ColumnType.CONTINUOUS,
    ),
    "optical_flow_radial_coherence_weighted_dt1": ColumnMetadata(
        name="Coherent Migration (Optical Flow Radial Coherence Weighted)",
        type=ColumnType.CONTINUOUS,
    ),
}
"""Mapping of column names to column metadata."""


COLUMN_METADATA_DICT: dict[str, dict[str, Any]] = {
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
        "label": "Cell Area",
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
        "label": "Mean Cell Edge Fluorescence",
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
