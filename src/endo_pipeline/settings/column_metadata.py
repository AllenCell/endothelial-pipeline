"""Column metadata structure and mapping to column names."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from numpy import pi

from endo_pipeline.io import slugify
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
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
    """Full column name in sentence case."""

    label: str | None = None
    """Short column label in sentence case. If not provided, set equal to name."""

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
        self.slug = slugify(self.name)

        # Set limits using minimum and maximum.
        self.limits = (self.min, self.max)


COLUMN_METADATA: dict[ColumnNameType, ColumnMetadata] = {
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
        name="Time under flow",
        label="Time",
        unit="hr",
        min="min",
        max="max",
        bin_width=0.5,
        ticks=range(0, 49, 12),
        type=ColumnType.CONTINUOUS,
    ),
    Column.TRACK_ID: ColumnMetadata(name="Track ID", type=ColumnType.DISCRETE),
    Column.TRACK_LENGTH: ColumnMetadata(name="Track duration", type=ColumnType.DISCRETE),
    Column.DATASET: ColumnMetadata(name="Dataset", type=ColumnType.DISCRETE),
    Column.POSITION: ColumnMetadata(name="Position", type=ColumnType.DISCRETE),
    Column.TIMEPOINT: ColumnMetadata(name="Frame number", type=ColumnType.DISCRETE),
    Column.CROP_INDEX: ColumnMetadata(name="Crop index", type=ColumnType.DISCRETE),
    Column.SegData.LABEL: ColumnMetadata(name="Segmentation label", type=ColumnType.DISCRETE),
    # Classic segmentation features ============================================
    Column.SegData.ALIGNMENT: ColumnMetadata(
        name="Alignment relative to flow",
        label="Cell alignment rel. to flow",
        unit="rad",
        min=0,
        max=pi / 2,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ALIGNMENT_DEG: ColumnMetadata(
        name="Alignment relative to flow",
        label="Cell alignment rel. to flow",
        unit="°",
        min=0,
        max=90,
        bin_width=1,
        ticks=range(0, 91, 15),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ORIENTATION: ColumnMetadata(
        name="Cell orientation",
        label="Cell orientation",
        unit="rad",
        min=0,
        max=pi,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ORIENTATION_DEG: ColumnMetadata(
        name="Cell orientation",
        label="Cell orientation",
        unit="°",
        min=0,
        max=180,
        bin_width=5,
        ticks=range(0, 181, 90),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NEMATIC_ORDER: ColumnMetadata(
        name="Nematic order",
        min=-1,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.AREA_UM_SQ: ColumnMetadata(
        name="Cell area",
        label="Cell area",
        unit=f"{Unicode.MU}m{Unicode.SQUARED}",
        min=350,
        max=2000,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ASPECT_RATIO: ColumnMetadata(
        name="Cell aspect ratio",
        label="Cell aspect ratio",
        min=1,
        max=10,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.PERIMETER_UM: ColumnMetadata(
        name="Cell perimeter",
        label="Cell perimeter",
        unit=f"{Unicode.MU}m",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.ECCENTRICITY: ColumnMetadata(
        name="Cell eccentricity",
        label="Cell eccentricity",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MAX: ColumnMetadata(
        name="Max VE-Cad fluorescence in cell",
        label="Cell fluorescence max",
        unit="a.u.",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MEAN: ColumnMetadata(
        name="Mean VE-Cad fluorescence in cell",
        label="Cell mean fluorescence",
        unit="a.u.",
        min=120,
        max=150,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CELL_FLUOR_MEDIAN: ColumnMetadata(
        name="Median VE-Cad fluorescence in cell",
        label="Cell median fluorescence",
        unit="a.u.",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.EDGE_FLUOR_MEAN: ColumnMetadata(
        name="Mean VE-Cad fluorescence at edges",
        label="Cell edge mean fluorescence",
        unit="a.u.",
        min=100,
        max=200,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NODE_FLUOR_MEAN: ColumnMetadata(
        name="Mean VE-Cad fluorescence at nodes",
        label="Cell node mean fluorescence",
        unit="a.u.",
        min=100,
        max=200,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.SOLIDITY: ColumnMetadata(
        name="Cell solidity",
        label="Cell solidity",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF: ColumnMetadata(
        name="Smoothed area difference (normalized)",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK: ColumnMetadata(
        name="Number of valid timepoints",
        type=ColumnType.DISCRETE,
    ),
    Column.SegData.NUM_NEIGHBORS: ColumnMetadata(
        name="Number of neighbors",
        label="Number of\nneighbors",
        min=0,
        max="max",
        bin_width=1,
        type=ColumnType.DISCRETE,
    ),
    Column.SegData.NUCLEI_POSITION_DISTANCE: ColumnMetadata(
        name="Nuclei-cell centroid distance",
        unit="px",
        min=0,
        max="max",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NUM_NUCLEI_AT_TIMEPOINT: ColumnMetadata(
        name="Number of nuclei",
        min=0,
        type=ColumnType.DISCRETE,
    ),
    Column.SegData.NUM_NUCLEI_IN_CROP: ColumnMetadata(
        name="Number of nuclei in patch",
        label="Number of nuclei in patch",
        min=0,
        type=ColumnType.DISCRETE,
    ),
    # Dynamic features =========================================================
    Column.SegData.CENTROID_VELOCITY_ANGLE_DEG: ColumnMetadata(
        name="Cell migration angle",
        label="Cell migration angle",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN: ColumnMetadata(
        name="Cell migration speed",
        label="Cell migration speed",
        unit=f"{Unicode.MU}m/min",
        min=0,
        max="max",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG: ColumnMetadata(
        name="Nucleus orientation relative to migration",
        label="Cell-nuc angle rel. migration",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD: ColumnMetadata(
        name="Cell-nucleus vs. migration dot product",
        label="Cell-nuc vs.\nmigration dot prod.",
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.NUCLEI_POSITION_ANGLE_DEG: ColumnMetadata(
        name="Nucleus orientation relative to flow angle",
        label="Cell-nuc angle rel. to flow",
        unit="°",
        min=-180,
        max=180,
        bin_width=5,
        ticks=range(-180, 181, 90),
        type=ColumnType.CONTINUOUS,
    ),
    Column.SegData.VECTOR_MEAN_FOR_CROP_MAGNITUDE: ColumnMetadata(
        name="Migration coherence in crop (vector mean magnitude)",
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
        name=f"Polar {Unicode.THETA}",
        label=Unicode.THETA,
        description="Polar angle calculated by transforming PC 1 and PC 2 to polar coordinates",
        min=0,
        max=pi,
        type=ColumnType.CONTINUOUS,
    ),
    Column.DiffAEData.POLAR_RADIUS: ColumnMetadata(
        name="Polar r",
        label="r",
        description="Polar radius calculated by transforming PC 1 and PC 2 to polar coordinates",
        min=0,
        type=ColumnType.CONTINUOUS,
    ),
    Column.DiffAEData.PC3_FLIPPED: ColumnMetadata(
        name=f"PC {Unicode.RHO}",
        label=Unicode.RHO,
        description="Negative value of PC 3",
        type=ColumnType.CONTINUOUS,
    ),
    Column.DiffAEData.NEMATIC_ORDER: ColumnMetadata(
        name="Nematic order parameter",
        label=f"cos(2{Unicode.THETA})",
        description="Nematic order calculated as cos(2*polar_theta)",
        min=-1,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    # Segmentation filters =====================================================
    Column.SegDataFilters.IS_EDGE_SEGMENTATION: ColumnMetadata(
        name="Filter: Touches edge of field of view",
        description="True if segmentation touches edge of FOV, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE: ColumnMetadata(
        name="Filter: Smoothed area change below threshold",
        description="True if smoothed area change is below threshold value, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION: ColumnMetadata(
        name="Filter: Exceeds min track duration",
        description="True if track duration is greater than minimum duration, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK: ColumnMetadata(
        name="Filter: Num valid points exceeds threshold",
        description="True if track has more points than minimum threshold value, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_INCLUDED: ColumnMetadata(
        name="Filter: Passed all filters",
        description="True if segmentation passes all filters, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegDataFilters.IS_VALID_BBOX: ColumnMetadata(
        name="Filter: Crop box limits are within FOV",
        description="True if crop bounding box are within the FOV, False otherwise",
        type=ColumnType.BOOLEAN,
    ),
    Column.SegData.NUM_TRACKS_BEFORE_FILTERING: ColumnMetadata(
        name="Number of segmentations before filtering",
        label="Num. segmentations\nbefore filtering",
        min=0,
        type=ColumnType.DISCRETE,
    ),
    Column.SegData.NUM_TRACKS_AFTER_FILTERING: ColumnMetadata(
        name="Number of segmentations after filtering",
        label="Num. segmentations\nafter filtering",
        min=0,
        type=ColumnType.DISCRETE,
    ),
    # Timepoint annotations ====================================================
    Column.Annotations.AUTO_BF_SCOPE_ERROR: ColumnMetadata(
        name="Filter: Auto-detected brightfield microscope error",
        description="Auto detected error with brightfield scope",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.AUTO_BF_TEMP_ARTIFACT: ColumnMetadata(
        name="Filter: Auto-detected temporary artifact",
        description="Auto detected temporary brightfield artifact.",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.AUTO_GFP_SCOPE_ERROR: ColumnMetadata(
        name="Filter: Auto-detected GFP channel microscope error",
        description="Auto detected error with GFP scope",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.BF_SCOPE_ERROR: ColumnMetadata(
        name="Filter: Manually annotated brightfield microscope error",
        description="Manually annotated error with brightfield scope",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.BF_TEMP_ARTIFACT: ColumnMetadata(
        name="Filter: Manually annotated temporary artifact",
        description="Manually annotated temporary brightfield artifact",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.GFP_SCOPE_ERROR: ColumnMetadata(
        name="Filter: Manually annotated GFP channel microscope error",
        description="Manually annotated error with GFP scope",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.CELL_PILING: ColumnMetadata(
        name="Filter: Manually annotated significant cell piling",
        description="Manually annotated range of timepoints where cells pile up (> 30% of FOV)",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.NOT_STEADY_STATE: ColumnMetadata(
        name="Filter: Cells not at steady state",
        description="Timepoint is not at visual steady state",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.UNFED: ColumnMetadata(
        name="Filter: Unfed (more than 3 hours since fresh media introduced)",
        description="Manually annotated timepoint where cells are more than 3hrs since last feeding",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.XY_SHIFT: ColumnMetadata(
        name="Filter: Significant change in XY position of FOV",
        description="Manually annotated shift in the XY position",
        type=ColumnType.BOOLEAN,
    ),
    Column.Annotations.Z_SHIFT: ColumnMetadata(
        name="Filter: Significant change in Z position of FOV",
        description="Manually annotated shift in the Z focus.",
        type=ColumnType.BOOLEAN,
    ),
    # Optical flow features ====================================================
    "optical_flow_mean_unit_vector_dt1": ColumnMetadata(
        name="Coherent migration (optical flow mean unit vector)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.ANGLE_MEAN: ColumnMetadata(
        name="Optical flow mean angle",
        unit="rad",
        min=0,
        max=8,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.ANGLE_STD: ColumnMetadata(
        name="Coherent migration (optical flow angle std dev)",
        min=0,
        max=4,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.SPEED_MEAN: ColumnMetadata(
        name="Optical flow mean speed",
        label="Patch-based\nmigration speed",
        unit="pixels/frame",
        min=0,
        max=8,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.SPEED_STD: ColumnMetadata(
        name="Optical flow speed std dev",
        min=0,
        max=10,
        type=ColumnType.CONTINUOUS,
    ),
    "optical_flow_mean_unit_vector_fast": ColumnMetadata(
        name="Coherent migration fast (optical flow unit vectors greater than 1 speed)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "speed_above_1_count": ColumnMetadata(
        name="N vectors with speed above 1",
        type=ColumnType.DISCRETE,
    ),
    "ema005_optical_flow_mean_unit_vector_dt1": ColumnMetadata(
        name="Coherent migration (EMA 0.05, optical flow mean unit vector)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "ema005_optical_flow_mean_unit_vector_fast_dt1": ColumnMetadata(
        name="Coherent migration (EMA 0.05, optical flow mean unit vector fast)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    Column.OpticalFlow.UNIT_VECTOR_MEAN: ColumnMetadata(
        name="Coherent migration (EMA 0.1, optical flow mean unit vector)",
        label="Patch-based\nmigration coherence",
        min=0,
        max=1,
        bin_width=0.02,
        type=ColumnType.CONTINUOUS,
    ),
    "ema01_optical_flow_mean_unit_vector_fast_dt1": ColumnMetadata(
        name="Coherent migration (EMA 0.1, optical flow mean unit vector fast)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "ema02_optical_flow_mean_unit_vector_dt1": ColumnMetadata(
        name="Coherent migration (EMA 0.2, optical flow mean unit vector)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "ema02_optical_flow_mean_unit_vector_fast_dt1": ColumnMetadata(
        name="Coherent migration (EMA 0.2, optical flow mean unit vector fast)",
        min=0,
        max=1,
        type=ColumnType.CONTINUOUS,
    ),
    "ema01_optical_flow_radial_coherence_dt1": ColumnMetadata(
        name="Coherent migration (EMA 0.1, optical flow radial coherence)",
        type=ColumnType.CONTINUOUS,
    ),
    "ema01_optical_flow_radial_coherence_weighted_dt1": ColumnMetadata(
        name="Coherent migration (EMA 0.1, optical flow radial coherence weighted)",
        type=ColumnType.CONTINUOUS,
    ),
    "optical_flow_radial_coherence_dt1": ColumnMetadata(
        name="Coherent migration (optical flow radial coherence)",
        type=ColumnType.CONTINUOUS,
    ),
    "optical_flow_radial_coherence_weighted_dt1": ColumnMetadata(
        name="Coherent migration (optical flow radial coherence weighted)",
        type=ColumnType.CONTINUOUS,
    ),
    Column.VectorField.PEARSON_R: ColumnMetadata(
        name="Pearson r of MFPT\n(grid versus track-based)",
        label="Pearson r",
        type=ColumnType.CONTINUOUS,
    ),
    Column.VectorField.LINEFIT_SLOPE: ColumnMetadata(
        name="Line fit slope",
        label="Line fit slope",
        type=ColumnType.CONTINUOUS,
    ),
}
"""Mapping of column names to column metadata."""
