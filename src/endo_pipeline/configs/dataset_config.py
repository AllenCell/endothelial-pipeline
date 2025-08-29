"""Data structures for dataset configs."""

from dataclasses import field
from typing import Literal

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass

MicroscopeType = Literal["3i", "Nikon"]
"""Valid microscope types."""

SampleType = Literal["live", "fixed", "fixed-methanol"]
"""Valid sample types."""

ObjectiveType = Literal["20X", "40X"]
"""Valid objective types."""


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
