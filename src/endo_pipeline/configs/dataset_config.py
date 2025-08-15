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

    flow: list
    """Flow conditions for the dataset."""

    n_total_positions: int
    """Total number of positions captured."""

    channel_488_index: int
    """Index of the 488 channel."""

    brightfield_channel_index: int
    """Index of the brightfield channel."""

    flow_conditions: list[FlowCondition] = field(default_factory=list)
    """List of flow conditions for the dataset."""

    channel_405_index: int | None = None
    """Index of the 405 channel."""

    channel_561_index: int | None = None
    """Index of the 561 channel."""

    channel_640_index: int | None = None
    """Index of the 640 channel."""

    valid_timepoints: ValidTimepoints | None = None
    """List of valid timepoint ranges. None if all timepoints are valid."""

    include_scenes: list[int] | None = None
    """List of scenes to include."""

    notes: str = ""
    """"Additional notes about dataset."""

    exclude_timepoints: dict[int, list[int]] = None
    """For each zarr position, manually annotated tps that should be dropped due to BF artifacts."""

    center_z_plane: dict[int, int] = None
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
