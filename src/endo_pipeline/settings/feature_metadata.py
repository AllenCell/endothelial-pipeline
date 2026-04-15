"""Feature metadata structure and mapping to column names."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from endo_pipeline.io import slugify
from endo_pipeline.settings.column_names import ColumnName as Column

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


FEATURE_METADATA_DICT = {
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
