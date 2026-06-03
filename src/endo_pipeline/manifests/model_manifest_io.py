"""Methods for model manifest I/O."""

import logging
from pathlib import Path

import yaml
from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from endo_pipeline.manifests import ModelManifest

logger = logging.getLogger(__name__)


def get_model_manifest_dir() -> Path:
    """Get path to model manifest directory."""

    return Path(__file__).resolve().parents[1] / "manifests" / "models"


def create_model_manifest(manifest_name: str, workflow_name: str | None = None) -> ModelManifest:
    """Create a new empty model manifest, or return one if it already exists."""

    manifest_dir = get_model_manifest_dir()
    manifest_file = manifest_dir / f"{manifest_name}.yaml"

    if manifest_file.exists():
        logger.warning("Model manifest [ %s ] already exists.", manifest_name)
        return load_model_manifest(manifest_name)
    else:
        return ModelManifest(name=manifest_name, workflow=Path(workflow_name or "").stem)


def load_model_manifest(manifest_name: str) -> ModelManifest:
    """Load model manifest by name."""

    manifest_dir = get_model_manifest_dir()
    manifest_file = manifest_dir / f"{manifest_name}.yaml"

    if not manifest_file.exists():
        logger.error("Model manifest [ %s ] could not be loaded", manifest_name)
        raise FileNotFoundError(f"No such file '{manifest_file}'")
    else:
        manifest = YAMLDecoder(ModelManifest).decode(manifest_file.read_text())
        logger.debug("Loaded model manifest [ %s ] from [ %s ]", manifest_name, manifest_file)
        return manifest


def save_model_manifest(manifest: ModelManifest) -> None:
    """Save model manifest to manifest directory."""

    manifest_dir = get_model_manifest_dir()
    manifest_file = manifest_dir / f"{manifest.name}.yaml"

    def yaml_encoder(data):  # type: ignore[no-untyped-def]
        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

    try:
        content = str(YAMLEncoder(ModelManifest, post_encoder_func=yaml_encoder).encode(manifest))
        manifest_file.write_text(content)
        logger.debug("Saved model manifest [ %s ] to [ %s ]", manifest.name, manifest_file)
    except:
        logger.error("Model manifest [ %s ] could not be saved", manifest.name)
        raise


def get_available_model_manifests() -> list[str]:
    """Get list of available model manifest names."""

    df_manifest_names = [path.stem for path in get_model_manifest_dir().iterdir()]
    logger.info("Available model manifests: %s", " | ".join(df_manifest_names))

    return df_manifest_names
