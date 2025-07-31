"""Methods for dataframe manifest I/O."""

import logging
from pathlib import Path

import yaml
from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from src.endo_pipeline.manifests import DataframeManifest

logger = logging.getLogger(__name__)


def get_dataframe_manifest_dir() -> Path:
    """Get path to dataframe manifest directory."""

    return Path(__file__).resolve().parents[1] / "manifests" / "dataframes"


def get_available_dataframe_manifest_names() -> list[str]:
    """Get list of available dataframe manifest names."""
    manifest_names = [path.stem for path in get_dataframe_manifest_dir().iterdir()]
    logger.info("Available dataframe manifests [ %s ]", " | ".join(manifest_names))
    return manifest_names


def load_dataframe_manifest(manifest_name: str) -> DataframeManifest:
    """Load dataframe manifest by name."""

    manifest_dir = get_dataframe_manifest_dir()
    manifest_file = manifest_dir / f"{manifest_name}.yaml"

    if not manifest_file.exists():
        logger.error("Dataframe manifest [ %s ] could not be loaded", manifest_name)
        raise FileNotFoundError(f"No such file '{manifest_file}'")
    else:
        manifest = YAMLDecoder(DataframeManifest).decode(manifest_file.read_text())
        logger.debug("Loaded dataframe manifest [ %s ] from [ %s ]", manifest_name, manifest_file)
        return manifest


def save_dataframe_manifest(manifest: DataframeManifest) -> None:
    """Save dataframe manifest to manifest directory."""

    manifest_dir = get_dataframe_manifest_dir()
    manifest_file = manifest_dir / f"{manifest.name}.yaml"

    def list_representer(dumper, data):
        # This representer saves lists as [a, b, c] unless it is a list of dicts.
        flow_style = not (len(data) > 0 and isinstance(data[0], dict))
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)

    def yaml_encoder(data):
        yaml.SafeDumper.add_representer(list, list_representer)
        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

    try:
        content = str(
            YAMLEncoder(DataframeManifest, post_encoder_func=yaml_encoder).encode(manifest)
        )
        manifest_file.write_text(content)
        logger.debug("Saved dataframe manifest [ %s ] to [ %s ]", manifest.name, manifest_file)
    except:
        logger.error("Dataframe manifest [ %s ] could not be saved", manifest.name)
        raise
