"""Data structures for dataset configs."""

from typing import Literal

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass


@dataclass
class ValidTimepoints:
    start: list[int]

    stop: list[int]


@dataclass
class DatasetConfig:
    """Dataset configuration for pipeline."""

    name: str
    """Unique name of the dataset."""

    original_path: str
    """Original path to dataset."""

    zarr_path: str
    """Path to dataset converted to Zarr format."""

    fmsid: str
    """FMS ID."""

    barcode: str
    """Dataset LabKey barcode."""

    cell_lines: list[str]
    """List of cell lines in dataset."""

    live_or_fixed_sample: Literal["live", "fixed", "fixed-methanol"]
    """Experimental condition that dataset was collected under."""

    microscope: Literal["3i", "Nikon"]
    """Microscope that dataset was collected with."""

    shear_stress_regime: str
    """Shear stress regime the dataset was collected under."""

    use_cases: list[
        Literal[
            "classic_segmentation",
            "feasibility",
            "immunofluorescence",
            "model_training",
            "measurement",
            "nuclear_label_free_predictions",
        ]
    ]
    """List of valid uses cases for the dataset."""

    pixel_size_xy_in_um: float
    """Pixel size in XY dimension in μm."""

    duration: int
    """Duration of dataset in frames."""

    time_interval_in_minutes: float
    """Time interval between frames in minutes."""

    flow: list[tuple[int, int, float]]
    """Flow conditions for the dataset."""

    n_total_positions: int
    """Total number of positions captured."""

    channel_488_index: int
    """Index of the 488 channel."""

    brightfield_channel_index: int
    """Index of the brightfield channel."""

    channel_405_index: int | None = None
    """Index of the 405 channel."""

    channel_561_index: int | None = None
    """Index of the 561 channel."""

    channel_640_index: int | None = None
    """Index of the 640 channel."""

    nuclear_label_free_seg_path: str | None = None
    """Path to nuclear label free segmentation."""

    nuclear_stain_seg_path: str | None = None
    """Path to nuclear stain segmentation."""

    nuclear_seg_manifest_fmsid: str | None = None
    """FMS ID for nuclear segmentation manifest."""

    diffae_manifest_fmsid: str | None = None
    """FMS ID for diffusion autoencoder manifest."""

    tracking_integration_fmsid: str | None = None
    """FMS ID for tracking integration."""

    diffae_tracking_integration_fmsid: str | None = None
    """FMS ID for diffusion autoencoder tracking integration."""

    immunofluorescence_manifest_fmsid: str | None = None
    """FMS ID for immunofluorescence manifest."""

    is_reference: bool = False
    """True if dataset is used as a reference dataset, False otherwise."""

    valid_timepoints: ValidTimepoints | None = None
    """List of valid timepoint ranges. None if all timepoints are valid."""

    cell_mean_features: str | None = None
    """FMS ID for cell mean features."""

    include_scenes: list[int] | None = None
    """List of scenes to include."""

    notes: str = ""
    """"Additional notes about dataset."""

    class Config(BaseConfig):
        forbid_extra_keys = True
        omit_none = False
