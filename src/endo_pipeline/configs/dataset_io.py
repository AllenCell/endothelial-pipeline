import logging
from pathlib import Path
from typing import Any

import yaml
from deprecated import deprecated  # type:ignore[import-untyped]

logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get path to the config directory."""

    return Path(__file__).resolve().parents[0]


def save_to_yaml(object: dict, path: Path, list_representer: bool = True) -> None:
    """Save dictionary object to YAML at given path."""

    if list_representer:
        yaml.SafeDumper.add_representer(
            list,
            lambda dumper, data: dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            ),
        )

    yaml_content = yaml.safe_dump(
        object, default_flow_style=False, sort_keys=False, width=80, indent=2
    )
    path.open("w").write(yaml_content)


def combine_data_config(save: bool = False) -> dict:
    """Combine individual dataset configs into combined dataset keyed by name."""

    separated_path = get_config_dir() / "datasets"
    combined_path = Path(__file__).resolve().parents[1] / "data_config.yaml"

    separate_data_configs = [
        yaml.safe_load(config.open()) for config in sorted(separated_path.glob("*.yaml"))
    ]
    combined_data_config = {config["name"]: config for config in separate_data_configs}

    if save:
        save_to_yaml(combined_data_config, combined_path)

    return combined_data_config


def combine_model_config(save: bool = False) -> dict:
    """Combine individual model configs into combined config keyed by name."""

    separated_path = get_config_dir() / "models"
    combined_path = Path(__file__).resolve().parents[1] / "model_config.yaml"

    separate_data_configs = [
        yaml.safe_load(config.open()) for config in sorted(separated_path.glob("*.yaml"))
    ]
    combined_model_config = {config["name"]: config for config in separate_data_configs}

    if save:
        save_to_yaml(combined_model_config, combined_path)

    return combined_model_config


# model methods


@deprecated(
    """
NOTE: you can ignore this warning when loading "dynamics" configs.

With the switch to loading dataset configs using the DatasetConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing dataset
configs is to use one of the following replacement methods. If you need configs
for all datasets, use:

        configs.load_all_dataset_configs

If you need the config for a single dataset, use:

        configs.load_dataset_config(dataset_name)

If you need only need dataset names, use:

        configs.get_available_dataset_names

With the switch to loading model configs using the ModelConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing model
configs is to use one of the following replacement methods.

If you need the config for a single model, use:

        configs.load_model_config(model_name)

If you need only need dataset names, use:

        configs.get_available_model_names
"""
)
def load_config(config_type: str = "data") -> dict[Any, Any]:
    """Load configuration from YAML file."""
    if config_type not in ["data", "model", "dynamics"]:
        raise ValueError('Invalid config type. Must be either "data", "model", or "dynamics."')

    # If loading the data config, load combined from all individual dataset
    # configs. This is part of a change to manage datasets with dataclasses.
    if config_type == "data":
        return combine_data_config()

    # If loading the model config, load combined from all individual model
    # configs. This is part of a change to manage models with dataclasses.
    if config_type == "model":
        return combine_model_config()

    config_dir = get_config_dir()
    config_file = config_dir / f"{config_type}_config.yaml"
    with open(config_file) as file:
        config_data = yaml.safe_load(file)
    return config_data
