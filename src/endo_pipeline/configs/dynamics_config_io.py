"""Methods for dataset config I/O."""

import logging
from pathlib import Path

import yaml
from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from src.endo_pipeline.configs import DynamicsConfig

logger = logging.getLogger(__name__)


def get_dynamics_config_dir() -> Path:
    """Get path to config directory."""

    return Path(__file__).resolve().parents[1] / "configs" / "dynamics_pipeline"


def get_available_dynamics_configs() -> list[str]:
    """Get list of available dataset names."""

    dynamics_config_names = [path.stem for path in get_dynamics_config_dir().iterdir()]
    logger.info(
        "Available configs for 2D Diff AE feature dynamics analysis [ %s ]",
        " | ".join(dynamics_config_names),
    )

    return dynamics_config_names


def validate_all_dynamics_configs() -> None:
    """Validate all dataset configs against defined schema."""

    dynamics_config_names = get_available_dynamics_configs()

    for dynamics_config_name in dynamics_config_names:
        validate_dynamics_config(dynamics_config_name)


def validate_dynamics_config(dynamics_config_name: str) -> None:
    """Validate given dataset config against defined schema."""

    config_dir = get_dynamics_config_dir()
    config_file = config_dir / f"{dynamics_config_name}.yaml"

    logger.info("Validating config file [ %s ]", dynamics_config_name)
    config = YAMLDecoder(DynamicsConfig).decode(config_file.read_text())

    if config.name != config_file.stem:
        logger.error(
            "Config file name [ %s ] does not match name field [ %s ]",
            config_file,
            config.name,
        )


def load_all_dynamics_configs() -> list[DynamicsConfig]:
    """Load all dataset configs."""

    dynamics_config_names = get_available_dynamics_configs()

    dynamics_configs = [load_dynamics_config(name) for name in dynamics_config_names]
    logger.info("Loaded all available datasets [ %s ]", " | ".join(dynamics_config_names))

    return dynamics_configs


def load_dynamics_config(dynamics_config_name: str) -> DynamicsConfig:
    """Load single dataset config by name."""

    config_dir = get_dynamics_config_dir()
    config_file = config_dir / f"{dynamics_config_name}.yaml"

    if not config_file.exists():
        logger.error("Dataset [ %s ] could not be loaded", dynamics_config_name)
        raise FileNotFoundError(f"No such file '{config_file}'")
    else:
        logger.debug("Loaded dataset [ %s ]", dynamics_config_name)
        return YAMLDecoder(DynamicsConfig).decode(config_file.read_text())


def save_dynamics_config(dynamics_config: DynamicsConfig) -> None:
    """Save dataset config to config directory."""

    config_dir = get_dynamics_config_dir()
    config_file = config_dir / f"{dynamics_config.name}.yaml"

    def list_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    def yaml_encoder(data):
        yaml.SafeDumper.add_representer(list, list_representer)
        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

    content = YAMLEncoder(DynamicsConfig, post_encoder_func=yaml_encoder).encode(dynamics_config)
    config_file.write_text(content)
