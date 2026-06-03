"""Methods for image manifest I/O."""

import logging
from pathlib import Path

import yaml
from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from endo_pipeline.manifests import ImageManifest

logger = logging.getLogger(__name__)


def get_image_manifest_dir() -> Path:
    """Get path to image manifest directory."""

    return Path(__file__).resolve().parents[1] / "manifests" / "images"


def create_image_manifest(manifest_name: str, workflow_name: str | None = None) -> ImageManifest:
    """Create a new empty image manifest, or return one if it already exists."""

    manifest_dir = get_image_manifest_dir()
    manifest_file = manifest_dir / f"{manifest_name}.yaml"

    if manifest_file.exists():
        logger.warning("Image manifest [ %s ] already exists.", manifest_name)
        return load_image_manifest(manifest_name)
    else:
        return ImageManifest(name=manifest_name, workflow=Path(workflow_name or "").stem)


def load_image_manifest(manifest_name: str) -> ImageManifest:
    """Load image manifest by name."""

    manifest_dir = get_image_manifest_dir()
    manifest_file = manifest_dir / f"{manifest_name}.yaml"

    if not manifest_file.exists():
        logger.error("Image manifest [ %s ] could not be loaded", manifest_name)
        raise FileNotFoundError(f"No such file '{manifest_file}'")
    else:
        manifest = YAMLDecoder(ImageManifest).decode(manifest_file.read_text())
        logger.debug("Loaded image manifest [ %s ] from [ %s ]", manifest_name, manifest_file)
        return manifest


def save_image_manifest(manifest: ImageManifest) -> None:
    """Save image manifest to manifest directory."""

    manifest_dir = get_image_manifest_dir()
    manifest_file = manifest_dir / f"{manifest.name}.yaml"

    def list_representer(dumper, data):
        # This representer saves lists as [a, b, c] unless it is a list of dicts.
        flow_style = not (len(data) > 0 and isinstance(data[0], dict))
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)

    def yaml_encoder(data):
        # Save copy of default representers
        default_representers = yaml.representer.Representer.yaml_representers.copy()

        # Override with custom representers
        yaml.SafeDumper.add_representer(list, list_representer)

        # Encode data into YAML
        encode = yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

        # Remove custom representers so they don't interfere with other uses
        yaml.SafeDumper.add_representer(list, default_representers[list])

        return encode

    try:
        content = str(YAMLEncoder(ImageManifest, post_encoder_func=yaml_encoder).encode(manifest))
        manifest_file.write_text(content)
        logger.debug("Saved image manifest [ %s ] to [ %s ]", manifest.name, manifest_file)
    except:
        logger.error("Image manifest [ %s ] could not be saved", manifest.name)
        raise


def get_available_image_manifests() -> list[str]:
    """Get list of available image manifest names."""

    df_manifest_names = [path.stem for path in get_image_manifest_dir().iterdir()]
    logger.info("Available image manifests: %s", " | ".join(df_manifest_names))

    return df_manifest_names
