"""Data structures for dataset configs."""

from dataclasses import field
from enum import Enum, StrEnum
from typing import Literal

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass

MicroscopeType = Literal["3i", "Nikon"]
"""Valid microscope types."""

SampleType = Literal["live", "fixed", "fixed-methanol"]
"""Valid sample types."""

ObjectiveType = Literal["20X", "40X"]
"""Valid objective types."""


class ShearStressRegime(Enum):
    """Shear stress regime categories with target shear stress ranges."""

    NO = ("no", 0.0, 0.0)
    """No shear stress."""

    MIN = ("min", 4.5, 7.2)
    """Minimum shear stress tested (target: 6 dyn/cm2)."""

    LOW = ("low", 8.5, 9.1)
    """Low shear stress (target: 9 dyn/cm2)"""

    MEDIUM = ("medium", 10.0, 12.5)
    """Medium shear stress (target: 12 dyn/cm2)."""

    HIGH = ("high", 13.0, 16.0)
    """High shear stress (target: 15 dyn/cm2)."""

    MAX = ("max", 18.5, 35.0)
    """Maximum shear stress tested (target: 20 dyn/cm2)."""

    upper: float
    """Upper bound of the shear stress regime."""

    lower: float
    """Lower bound of the shear stress regime."""

    def __new__(cls, value: str, lower: float, upper: float) -> "ShearStressRegime":
        """Create a new shear stress regime."""

        obj = object.__new__(cls)
        obj._value_ = value
        obj.lower = lower
        obj.upper = upper

        return obj


class ChannelName(StrEnum):
    """Standardized names for channels."""

    BF = "Brightfield"
    """Channel name for brightfield."""


class TimepointAnnotation(StrEnum):
    """Annotations for timepoints that should be excluded from model training and/or analysis."""

    AUTO_BF_SCOPE_ERROR = "auto_bf_scope_error"
    """Auto detected error with brightfield scope."""

    AUTO_BF_TEMP_ARTIFACT = "auto_bf_temp_artifact"
    """Auto detected Temporary brightfield artifact."""

    AUTO_GFP_SCOPE_ERROR = "auto_gfp_scope_error"
    """Auto detected error with GFP scope."""

    BF_SCOPE_ERROR = "bf_scope_error"
    """Manually annotated error with brightfield scope."""

    BF_TEMP_ARTIFACT = "bf_temp_artifact"
    """Manually Temporary brightfield artifact."""

    CELL_PILING = "cell_piling"
    """Manually annotated range of timepoints where cells pile up (> 30% of FOV)."""

    GFP_SCOPE_ERROR = "gfp_scope_error"
    """Manually annotated error with GFP scope."""

    NOT_STEADY_STATE = "not_steady_state"
    """Timepoint is not at visual steady state."""

    UNFED = "unfed"
    """Manually annotated timepoint where cells are more than 3hrs since last feeding."""

    XY_SHIFT = "xy_shift"
    """Manually annotated shift in the XY position."""

    Z_SHIFT = "z_shift"
    """Manually annotated shift in the Z focus."""


class PositionAnnotation(StrEnum):
    """Annotations for positions that should be excluded from model training and/or analysis."""

    DUST_ARTIFACT = "dust_artifact"
    """Manually annotated position includes a dust artifact."""

    HOLE_IN_MONOLAYER = "hole_in_monolayer"
    """Hole in the monolayer that persist throughout the entire duration of timelapse."""


@dataclass
class FlowCondition:
    """
    Flow condition for a dataset. Negative start or stop frames indicate the flow occurred prior
    to image acquisition. Time is represented in 5 minute intervals, even if the time was not
    during an acquisition, so it wont be more than ~2.5 minutes off from the actual time.
    """

    start: int
    """Start frame of flow condition."""

    stop: int
    """Stop frame of flow condition."""

    shear_stress: float
    """Shear stress in dynes/cm^2 for the flow condition."""

    shear_stress_bin: int = field(init=False)
    """Shear stress bin (bin size of 3)."""

    def __post_init__(self):
        """Post initialization steps for flow condition."""

        # Bin shear stress with bin size of 3. Note that round uses bankers
        # rounding, which rounds to the nearest even number in order to average
        # out rounding errors.
        self.shear_stress_bin = 3 * round(self.shear_stress / 3)


@dataclass
class ChannelIndices:
    """Indices of individual channels."""

    brightfield: int
    """Index of the brightfield channel."""

    channel_488: int
    """Index of the 488 channel."""

    channel_405: int | None = None
    """Index of the 405 channel."""

    channel_561: int | None = None
    """Index of the 561 channel."""

    channel_640: int | None = None
    """Index of the 640 channel."""


@dataclass
class DatasetConfig:
    """Dataset configuration for pipeline."""

    name: str
    """Unique name of the dataset."""

    date: str
    """Date the dataset was collected, formatted as YYYYMMDD."""

    original_path: str
    """Original path to dataset."""

    zarr_positions: list[int]
    """List of available Zarr positions."""

    fmsid: str
    """FMS ID."""

    barcode: str
    """Dataset LabKey barcode."""

    cell_lines: list[str]
    """List of cell lines in dataset."""

    live_or_fixed_sample: SampleType
    """Experimental condition that dataset was collected under."""

    is_timelapse: bool
    """True if dataset is a timelapse dataset, False otherwise."""

    microscope: MicroscopeType
    """Microscope that dataset was collected with."""

    objective: ObjectiveType
    """Objective that dataset was collected under."""

    shear_stress_regime: list[ShearStressRegime]
    """Shear stress regime the dataset was collected under."""

    pixel_size_xy_in_um: float
    """Pixel size in XY dimension in μm."""

    duration: int
    """Duration of dataset in frames."""

    time_interval_in_minutes: float | None
    """Time interval between frames in minutes."""

    n_total_positions: int
    """Total number of positions captured."""

    original_channel_indices: ChannelIndices
    """Channel indices for original dataset."""

    zarr_channel_indices: ChannelIndices
    """Channel indices for dataset converted to Zarr format."""

    channel_names: list[str | ChannelName]
    """List of channel names."""

    flow_conditions: list[FlowCondition]
    """List of flow conditions for the dataset."""

    include_scenes: list[int] | None = None
    """List of scenes to include."""

    notes: str = ""
    """"Additional notes about dataset."""

    timepoint_annotations: (
        dict[TimepointAnnotation, dict[int, list[int | tuple[int, int]]]] | None
    ) = None
    """Manually annotated timepoints per position. Individual tps (int) or start, stops (tuple)."""

    position_annotations: dict[PositionAnnotation, list[int]] | None = None
    """Manually annotated positions."""

    center_z_plane: dict[int, int] | None = None
    """For each zarr position, the calculated and visually validated center Z-plane"""

    class Config(BaseConfig):
        """Settings for dataset config."""

        forbid_extra_keys = True
        omit_none = False


@dataclass
class DatasetCollectionConfig:
    """Dataset configuration collection for pipeline."""

    name: str
    """Unique name of the dataset collection."""

    description: str
    """Brief description of the dataset collection."""

    datasets: list[str]
    """List of dataset names that belong in the collection."""
