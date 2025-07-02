"""Methods for model config I/O."""

import logging
from pathlib import Path

import yaml
from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from src.endo_pipeline.configs import ModelConfig

logger = logging.getLogger(__name__)


def get_model_config_dir() -> Path:
    """Get path to config directory."""

    return Path(__file__).resolve().parents[1] / "configs" / "models"


def get_available_model_names() -> list[str]:
    """Get list of available model names."""

    model_names = [path.stem for path in get_model_config_dir().iterdir()]
    logger.info("Available models [ %s ]", " | ".join(model_names))

    return model_names


def validate_all_model_configs() -> None:
    """Validate all model configs against defined schema."""

    model_names = get_available_model_names()

    for model_name in model_names:
        validate_single_model_config(model_name)


def validate_single_model_config(model_name: str) -> None:
    """Validate given model config against defined schema."""

    config_dir = get_model_config_dir()
    config_file = config_dir / f"{model_name}.yaml"

    logger.info("Validating config file [ %s ]", model_name)
    config = YAMLDecoder(ModelConfig).decode(config_file.read_text())

    if config.name != config_file.stem:
        logger.error(
            "Config file name [ %s ] does not match name field [ %s ]",
            config_file,
            config.name,
        )


def load_all_model_configs() -> list[ModelConfig]:
    """Load all model configs."""

    model_names = get_available_model_names()

    models = [load_model_config(name) for name in model_names]
    logger.info("Loaded all available models [ %s ]", " | ".join(model_names))

    return models


def load_model_config(model_name: str) -> ModelConfig:
    """Load single model config by name."""

    config_dir = get_model_config_dir()
    config_file = config_dir / f"{model_name}.yaml"

    if not config_file.exists():
        logger.error("Model [ %s ] could not be loaded", model_name)
        raise FileNotFoundError(f"No such file '{config_file}'")
    else:
        logger.debug("Loaded model [ %s ]", model_name)
        return YAMLDecoder(ModelConfig).decode(config_file.read_text())


def save_model_config(model: ModelConfig) -> None:
    """Save model config to config directory."""

    config_dir = get_model_config_dir()
    config_file = config_dir / f"{model.name}.yaml"

    def yaml_encoder(data):
        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

    content = YAMLEncoder(ModelConfig, post_encoder_func=yaml_encoder).encode(model)
    config_file.write_text(content)


if __name__ == "__main__":
    validate_all_model_configs()
