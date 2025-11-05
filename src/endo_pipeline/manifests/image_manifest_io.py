"""Methods for image manifest I/O."""

import logging
from pathlib import Path

import yaml
from mashumaro.codecs.yaml import YAMLDecoder, YAMLEncoder

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.manifests import ImageLocation, ImageManifest

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
        yaml.SafeDumper.add_representer(list, list_representer)
        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)

    try:
        content = str(YAMLEncoder(ImageManifest, post_encoder_func=yaml_encoder).encode(manifest))
        manifest_file.write_text(content)
        logger.debug("Saved image manifest [ %s ] to [ %s ]", manifest.name, manifest_file)
    except:
        logger.error("Image manifest [ %s ] could not be saved", manifest.name)
        raise


def add_image_location_to_manifest(
    dataset_config: DatasetConfig,
    manifest_name: str,
    path_prefix: str | Path,
    add_directory: bool = False,
) -> None:
    """
    Add or update image location for given dataset in the manifest

    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration object
    manifest_name : str
        Name of the image manifest
    add_directory : bool
        Whether to add a directory structure to the path with date and fmsid
    path_prefix : str | Path
        Prefix path to the image location
    """

    img_manifest = load_image_manifest(manifest_name)

    date = dataset_config.name[:8]
    fmsid = dataset_config.fmsid
    suffix = "P{{position}}.ome.zarr"

    if add_directory:
        suffix = f"/{date}_{fmsid}/{date}_{fmsid}_{suffix}"
    else:
        suffix = f"{date}_{fmsid}_{suffix}"

    new_path = f"{path_prefix}/{suffix}"
    img_manifest.locations[dataset_config.name] = ImageLocation(path=Path(new_path))

    save_image_manifest(img_manifest)
