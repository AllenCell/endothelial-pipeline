"""Methods for dataset config I/O."""

import logging
from pathlib import Path

from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from src.endo_pipeline.configs import DatasetConfig
from src.endo_pipeline.configs.config_io import yaml_encoder

logger = logging.getLogger(__name__)


def get_dataset_config_dir() -> Path:
    """Get path to config directory."""

    return Path(__file__).resolve().parents[1] / "configs" / "datasets"


def get_available_dataset_names() -> list[str]:
    """Get list of available dataset names."""

    dataset_names = [path.stem for path in get_dataset_config_dir().iterdir()]
    logger.info("Available datasets [ %s ]", " | ".join(dataset_names))

    return dataset_names


def validate_all_dataset_configs() -> None:
    """Validate all dataset configs against defined schema."""

    dataset_names = get_available_dataset_names()

    for dataset_name in dataset_names:
        validate_single_dataset_config(dataset_name)


def validate_single_dataset_config(dataset_name: str) -> None:
    """Validate given dataset config against defined schema."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset_name}.yaml"

    logger.info("Validating config file [ %s ]", dataset_name)
    config = YAMLDecoder(DatasetConfig).decode(config_file.read_text())

    if config.name != config_file.stem:
        logger.error(
            "Config file name [ %s ] does not match name field [ %s ]",
            config_file,
            config.name,
        )


def load_all_dataset_configs() -> list[DatasetConfig]:
    """Load all dataset configs."""

    dataset_names = get_available_dataset_names()

    datasets = [load_single_dataset_config(name) for name in dataset_names]
    logger.info("Loaded all available datasets [ %s ]", " | ".join(dataset_names))

    return datasets


def load_reference_dataset_configs() -> list[DatasetConfig]:
    """Load all reference dataset configs."""

    all_datasets = load_all_dataset_configs()
    reference_datasets = [dataset for dataset in all_datasets if dataset.is_reference]

    reference_dataset_names = [dataset.name for dataset in reference_datasets]
    logger.info("Loaded all reference datasets [ %s ]", " | ".join(reference_dataset_names))

    return reference_datasets


def load_single_dataset_config(dataset_name: str) -> DatasetConfig:
    """Load single dataset config by name."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset_name}.yaml"

    if not config_file.exists():
        logger.error("Dataset [ %s ] could not be loaded", dataset_name)
        raise FileNotFoundError(f"No such file '{config_file}'")
    else:
        logger.debug("Loaded dataset [ %s ]", dataset_name)
        return YAMLDecoder(DatasetConfig).decode(config_file.read_text())


def save_dataset_config(dataset: DatasetConfig) -> None:
    """Save dataset config to config directory."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset.name}.yaml"

    content = YAMLEncoder(DatasetConfig, post_encoder_func=yaml_encoder).encode(dataset)
    config_file.write_text(content)


if __name__ == "__main__":
    validate_all_dataset_configs()
