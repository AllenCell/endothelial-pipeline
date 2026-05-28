"""Methods for dataset config I/O."""

import logging
import re
from pathlib import Path

import yaml
from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from endo_pipeline.configs import DatasetCollectionConfig, DatasetConfig

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
    logger.debug("Available datasets [ %s ]", " | ".join(dataset_names))

    return dataset_names


def get_available_dataset_collection_names() -> list[str]:
    """Get list of available dataset collection names."""

    collection_names = [path.stem for path in get_dataset_collection_config_dir().iterdir()]
    logger.debug("Available dataset collections [ %s ]", " | ".join(collection_names))

    return collection_names


def validate_all_dataset_configs() -> None:
    """Validate all dataset configs against defined schema."""

    dataset_names = get_available_dataset_names()

    for dataset_name in dataset_names:
        validate_dataset_config(dataset_name)


def validate_dataset_config(dataset_name: str) -> None:
    """Validate given dataset config against defined schema."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset_name}.yaml"

    logger.debug("Validating dataset config file [ %s ]", dataset_name)
    config = load_dataset_config(dataset_name)

    if config.name != config_file.stem:
        logger.error(
            "Config file name [ %s ] does not match name field [ %s ]",
            config_file,
            config.name,
        )

    if not config.is_timelapse:
        if config.duration != 1:
            logger.error(
                "Validation failed for dataset [ %s ]: "
                "If is_timelapse is False, duration must be equal to 1.",
                dataset_name,
            )
        if config.time_interval_in_minutes is not None:
            logger.error(
                "Validation failed for dataset [ %s ]: "
                "If is_timelapse is False, time_interval_in_minutes must be None.",
                dataset_name,
            )
    else:
        if config.duration <= 1:
            logger.error(
                "Validation failed for dataset [ %s ]: "
                "If is_timelapse is True, duration must be greater than 1.",
                dataset_name,
            )
        if config.time_interval_in_minutes is None:
            logger.error(
                "Validation failed for dataset [ %s ]: "
                "If is_timelapse is True, time_interval_in_minutes must not be None.",
                dataset_name,
            )

    for regime, flow in zip(config.shear_stress_regime, config.flow_conditions, strict=False):
        if flow.shear_stress < regime.lower or flow.shear_stress > regime.upper:
            logger.error(
                "Validation failed for dataset [ %s ]: "
                "Shear stress [ %d ] outside range for regime [ %s (%d - %d) ]",
                dataset_name,
                flow.shear_stress,
                regime.value,
                regime.lower,
                regime.upper,
            )


def load_all_dataset_configs() -> list[DatasetConfig]:
    """Load all dataset configs."""

    dataset_names = get_available_dataset_names()

    datasets = [load_dataset_config(name) for name in dataset_names]
    logger.debug("Loaded all available datasets [ %s ]", " | ".join(dataset_names))

    return datasets


def load_dataset_config(dataset_name: str) -> DatasetConfig:
    """Load dataset config by name."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset_name}.yaml"

    if not config_file.exists():
        logger.error("Dataset config [ %s ] could not be loaded", dataset_name)
        raise FileNotFoundError(f"No such file '{config_file}'")
    else:
        config_text = config_file.read_text()

        # Custom adjustment to split the shear stress regime into list.
        replace, regime = re.findall(r"(shear_stress_regime: (['a-z_]+))", config_text)[0]
        config_text = config_text.replace(
            replace, f"shear_stress_regime: [{ ','.join(regime.split('_to_')) }]"
        )

        config = YAMLDecoder(DatasetConfig).decode(config_text)

        # Log warning if any of the shear stress bins are more than +/- 1 away
        # from the center of the bin
        for flow_condition in config.flow_conditions:
            delta = abs(flow_condition.shear_stress - flow_condition.shear_stress_bin)
            if delta > 1:
                logger.warning(
                    "Dataset '%s' shear stress '%.2f' binned to '%d' (delta = '%.2f')",
                    dataset_name,
                    flow_condition.shear_stress,
                    flow_condition.shear_stress_bin,
                    delta,
                )

        logger.debug("Loaded dataset config [ %s ] from [ %s ]", dataset_name, config_file)
        return config


def save_dataset_config(dataset: DatasetConfig) -> None:
    """Save dataset config to config directory."""

    config_dir = get_dataset_config_dir()
    config_file = config_dir / f"{dataset.name}.yaml"

    def list_representer(dumper, data):
        # This representer saves lists as [a, b, c] unless it is a list of dicts.
        flow_style = not (len(data) > 0 and isinstance(data[0], dict))
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)

    def dict_representer(dumper, data):
        # This representer saves dict with ordered keys only if it is a dict of dicts.
        if isinstance(data[next(iter(data))], dict):
            return dumper.represent_dict(dict(sorted(data.items())))
        return dumper.represent_dict(data)

    def yaml_encoder(data):
        # Save copy of default representers
        default_representers = yaml.representer.Representer.yaml_representers.copy()

        # Override with custom representers
        yaml.SafeDumper.add_representer(list, list_representer)
        yaml.SafeDumper.add_representer(dict, dict_representer)

        # Custom adjustment to combine shear stress regime into single string
        data["shear_stress_regime"] = "_to_".join(data["shear_stress_regime"])

        # Custom adjustment to drop shear stress bin
        for condition in data["flow_conditions"]:
            del condition["shear_stress_bin"]

        # Encode data into YAML
        encode = yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

        # Remove custom representers so they don't interfere with other uses
        yaml.SafeDumper.add_representer(list, default_representers[list])
        yaml.SafeDumper.add_representer(dict, default_representers[dict])

        return encode

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


def get_datasets_in_collection(collection_name: str, subset: list[str] | None = None) -> list[str]:
    """Get list of dataset names in given collection."""

    collection = load_dataset_collection_config(collection_name)
    datasets = collection.datasets

    # Optional filtering of dataset names based on provided subset. Only dataset
    # names in both the collection and the subset are returned.
    if subset is not None:
        datasets = [name for name in datasets if name in subset]

    return datasets
