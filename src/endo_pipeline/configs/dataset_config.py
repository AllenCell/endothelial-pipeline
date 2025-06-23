import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.config import BaseConfig

logger = logging.getLogger(__name__)


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

    brightfield_channel_index: int
    """Index of the brightfield channel."""

    nuclear_label_free_seg_path: str
    """Path to nuclear label free segmentation."""

    nuclear_stain_seg_path: str | None = None
    """Path to nuclear stain segmentation."""

    channel_488_index: int | None = None
    """Index of the 488 channel."""

    channel_561_index: int | None = None
    """Index of the 561 channel."""

    channel_640_index: int | None = None
    """Index of the 640 channel."""

    channel_405_index: int | None = None
    """Index of the 405 channel."""

    nuclear_seg_manifest_fmsid: str | None = None
    """FMS ID for nuclear segmentation manifest."""

    diffae_manifest_fmsid: str | None = None
    """FMS ID for diffusion autoencoder manifest."""

    tracking_integration_fmsid: str | None = None
    """FMS ID for tracking integration."""

    diffae_tracking_integration_fmsid: str | None = None
    """FMS ID for diffusion autoencoder tracking integration."""

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


def get_config_dir() -> Path:
    """Get path to config directory."""

    return Path(__file__).resolve().parents[1] / "configs"


def get_available_datasets() -> list[str]:
    """Get list of available dataset names."""

    dataset_names = [path.stem for path in (get_config_dir() / "datasets").iterdir()]
    logger.info("Available datasets [ %s ]", " | ".join(dataset_names))

    return dataset_names


def validate_all_dataset_configs() -> None:
    """Validate all dataset configs against defined schema."""

    dataset_names = get_available_datasets()

    for dataset_name in dataset_names:
        validate_single_dataset_config(dataset_name)


def validate_single_dataset_config(dataset_name: str) -> None:
    """Validate given dataset config against defined schema."""

    config_dir = get_config_dir()
    config_file = config_dir / "datasets" / f"{dataset_name}.yaml"

    logger.info("Validating config file [ %s ]", dataset_name)
    config = YAMLDecoder(DatasetConfig).decode(config_file.read_text())

    if config.name != config_file.stem:
        logger.error(
            "Config file name [ %s ] does not match name field [ %s ]",
            config_file,
            config.name,
        )


def load_all_datasets() -> list[DatasetConfig]:
    """Load all dataset configs."""

    dataset_names = get_available_datasets()

    datasets = [load_single_dataset(name) for name in dataset_names]
    logger.info("Loaded all available datasets [ %s ]", " | ".join(dataset_names))

    return datasets


def load_reference_datasets() -> list[DatasetConfig]:
    """Load all reference dataset configs."""

    all_datasets = load_all_datasets()
    reference_datasets = [dataset for dataset in all_datasets if dataset.is_reference]

    reference_dataset_names = [dataset.name for dataset in reference_datasets]
    logger.info("Loaded all reference datasets [ %s ]", " | ".join(reference_dataset_names))

    return reference_datasets


def load_single_dataset(dataset_name: str) -> DatasetConfig | None:
    """Load single dataset config by name."""

    config_dir = get_config_dir()
    config_file = config_dir / "datasets" / f"{dataset_name}.yaml"

    if not config_file.exists():
        logger.warning(
            "Dataset [ %s ] not found at config directory [ %s ]", dataset_name, config_dir
        )
        return None
    else:
        logger.info("Loaded dataset [ %s ]", dataset_name)
        return YAMLDecoder(DatasetConfig).decode(config_file.read_text())


if __name__ == "__main__":
    validate_all_dataset_configs()
