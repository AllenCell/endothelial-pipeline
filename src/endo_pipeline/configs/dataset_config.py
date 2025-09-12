"""Data structures for dataset configs."""

from enum import StrEnum
from typing import Literal

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass

MicroscopeType = Literal["3i", "Nikon"]
"""Valid microscope types."""

SampleType = Literal["live", "fixed", "fixed-methanol"]
"""Valid sample types."""

ObjectiveType = Literal["20X", "40X"]
"""Valid objective types."""


class TimepointAnnotation(StrEnum):
    """Annotations for timepoints that should be excluded from model training and/or analysis."""

    AUTO_BF_SCOPE_ERROR = "auto_bf_scope_error"
    """Auto detected error with brightfield scope."""

    AUTO_BF_TEMP_ARTIFACT = "auto_bf_temp_artifact"
    """Auto detected Temporary brightfield artifact."""

    BF_SCOPE_ERROR = "bf_scope_error"
    """Manually annotated error with brightfield scope."""

    BF_TEMP_ARTIFACT = "bf_temp_artifact"
    """Manually Temporary brightfield artifact."""

    CELL_PILING = "cell_piling"
    """Manually annotated range of timepoints where cells pile up (> 30% of FOV)."""

    GFP_SCOPE_ERROR = "gfp_scope_error"
    """Manually annotated error with GFP scope."""

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


@dataclass
class ValidTimepoints:
    """
    Timepoints that are visually validated to be after steady state from no flow to a set
    flow condition appears to have stabilized and before cell piling occurs.
    """

    start: list[int]
    """Start frame of valid timepoints."""

    stop: list[int]
    """Stop frame of valid timepoints."""


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

    original_path: str
    """Original path to dataset."""

    zarr_path: str
    """Path to dataset converted to Zarr format."""

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

    shear_stress_regime: str
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

    flow_conditions: list[FlowCondition]
    """List of flow conditions for the dataset."""

    valid_timepoints: ValidTimepoints | None = None
    """List of valid timepoint ranges. None if all timepoints are valid."""

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
