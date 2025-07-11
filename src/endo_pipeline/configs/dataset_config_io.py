"""Methods for dataset config I/O."""

import logging
from pathlib import Path

import yaml
from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from src.endo_pipeline.configs import DatasetCollectionConfig, DatasetConfig

logger = logging.getLogger(__name__)


def get_dataset_config_dir() -> Path:
    """Get path to dataset config directory."""

    return Path(__file__).resolve().parents[1] / "configs" / "datasets"


def get_dataset_collection_config_dir() -> Path:
    """Get path to dataset collection config directory."""

    return Path(__file__).resolve().parents[1] / "configs" / "collections"


def get_available_dataset_names() -> list[str]:
    """Get list of available dataset names."""

    dataset_names = [path.stem for path in get_dataset_config_dir().iterdir()]
    logger.info("Available datasets [ %s ]", " | ".join(dataset_names))

    return dataset_names


def validate_all_dataset_configs() -> None:
    """Validate all dataset configs against defined schema."""

    dataset_names = get_available_dataset_names()

    for dataset_name in dataset_names:
        validate_dataset_config(dataset_name)


def validate_dataset_config(dataset_name: str) -> None:
    """Validate given dataset config against defined schema."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset_name}.yaml"

    logger.info("Validating dataset config file [ %s ]", dataset_name)
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

    datasets = [load_dataset_config(name) for name in dataset_names]
    logger.info("Loaded all available datasets [ %s ]", " | ".join(dataset_names))

    return datasets


def load_dataset_config(dataset_name: str) -> DatasetConfig:
    """Load dataset config by name."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset_name}.yaml"

    if not config_file.exists():
        logger.error("Dataset config [ %s ] could not be loaded", dataset_name)
        raise FileNotFoundError(f"No such file '{config_file}'")
    else:
        config = YAMLDecoder(DatasetConfig).decode(config_file.read_text())
        logger.debug("Loaded dataset config [ %s ] from [ %s ]", dataset_name, config_file)
        return config


def save_dataset_config(dataset: DatasetConfig) -> None:
    """Save dataset config to config directory."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset.name}.yaml"

    def list_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    def yaml_encoder(data):
        yaml.SafeDumper.add_representer(list, list_representer)
        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

    try:
        content = str(YAMLEncoder(DatasetConfig, post_encoder_func=yaml_encoder).encode(dataset))
        config_file.write_text(content)
        logger.debug("Saved dataset config [ %s ] to [ %s ]", dataset.name, config_file)
    except:
        logger.error("Dataset config [ %s ] could not be saved", dataset.name)
        raise


def load_dataset_collection_config(collection_name: str) -> DatasetCollectionConfig:
    """Load dataset collection config by name."""

    collection_dir = get_dataset_collection_config_dir()
    collection_file = collection_dir / f"{collection_name}.yaml"

    if not collection_file.exists():
        logger.error("Dataset collection config [ %s ] could not be loaded", collection_name)
        raise FileNotFoundError(f"No such file '{collection_file}'")
    else:
        logger.debug(
            "Loaded dataset collection config [ %s ] from [ %s ]", collection_name, collection_file
        )
        return YAMLDecoder(DatasetCollectionConfig).decode(collection_file.read_text())


def save_dataset_collection_config(collection: DatasetCollectionConfig) -> None:
    """Save dataset collection config to config directory."""

    collection_dir = get_dataset_collection_config_dir()
    collection_file = collection_dir / f"{collection.name}.yaml"

    def yaml_encoder(data):
        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

    try:
        content = str(
            YAMLEncoder(DatasetCollectionConfig, post_encoder_func=yaml_encoder).encode(collection)
        )
        collection_file.write_text(content)
        logger.debug(
            "Saved dataset collection config [ %s ] to [ %s ]", collection.name, collection_file
        )
    except:
        logger.error("Dataset collection config [ %s ] could not be saved", collection.name)
        raise


def get_datasets_in_collection(collection_name: str) -> list[str]:
    """Get list of dataset names in given collection."""

    collection = load_dataset_collection_config(collection_name)
    return collection.datasets
