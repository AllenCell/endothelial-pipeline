"""Methods for loading models."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from cyto_dl.api import CytoDLModel
    from cyto_dl.models.im2im.diffusion_autoencoder import (
        DiffusionAutoEncoder as BaseDiffusionAutoEncoder,
    )

    from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder

from endo_pipeline.manifests import ModelLocation

logger = logging.getLogger(__name__)


def load_model_from_path(
    path: Path | tuple[Path, Path], *, instantiate: bool = False
) -> "CytoDLModel | BaseDiffusionAutoEncoder | DiffusionAutoEncoder | CellposeModel":
    """
    Load model from path.

    The model type is determined by the type of path:

    - single path is assumed to be a Cellpose model
    - tuple of path (checkpoint, config) is assumed to be a CytoDL model

    Parameters
    ----------
    path
       Paths for model checkpoint and (optionally) config files.
    instantiate
        True to instantiate the model object, False otherwise.

    Returns
    -------
    :
        Model loaded from path.
    """

    if isinstance(path, Path):
        return load_cellpose_model_from_path(path)
    elif isinstance(path, tuple):
        return load_cytodl_model_from_path(path[0], path[1], instantiate=instantiate)

    logger.error("Argument '%s' must be of type 'Path' or 'tuple[Path, Path]'", path)
    raise ValueError("Unable to determine model from path type.")


def load_model_from_fms(
    fmsid: str | tuple[str, str], *, instantiate: bool = False
) -> "CytoDLModel | BaseDiffusionAutoEncoder | DiffusionAutoEncoder | CellposeModel":
    """
    Load model from FMS by file ID.

    The model type is determined by the type of path:

    - single file ID is assumed to be a Cellpose model
    - tuple of file ID (checkpoint and config) is assumed to be a CytoDL model

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `aicsfiles` installed.

    Parameters
    ----------
    fmsid
        FMS file ID for model checkpoint and (optionally) config files.
    instantiate
        True to instantiate the model object, False otherwise.

    Returns
    -------
    :
        Model loaded from FMS.
    """

    from endo_pipeline.io.fms import get_local_path_from_fmsid

    if isinstance(fmsid, str):
        checkpoint_path = get_local_path_from_fmsid(fmsid)
        return load_cellpose_model_from_path(checkpoint_path)
    elif isinstance(fmsid, tuple):
        checkpoint_path = get_local_path_from_fmsid(fmsid[0])
        config_path = get_local_path_from_fmsid(fmsid[1])
        return load_cytodl_model_from_path(checkpoint_path, config_path, instantiate=instantiate)

    logger.error("Argument '%s' must be of type 'str' or 'tuple[str, str]'", fmsid)
    raise ValueError("Unable to determine model from fmsid type.")


def load_model_from_s3(
    s3uri: str | tuple[str, str], *, instantiate: bool = False
) -> "CytoDLModel | BaseDiffusionAutoEncoder | DiffusionAutoEncoder | CellposeModel":
    """
    Load model from S3 by object URI.

    The model type is determined by the type of path:

    - single object URI is assumed to be a Cellpose model
    - tuple of object URI (checkpoint and config) is assumed to be a CytoDL model

    Parameters
    ----------
    s3uri
        S3 object URI for model checkpoint and (optionally) config files.
    instantiate
        True to instantiate the model object, False otherwise.

    Returns
    -------
    :
        Model loaded from S3.
    """

    from endo_pipeline.io.aws import download_s3_file_to_path

    if isinstance(s3uri, str):
        checkpoint_path = download_s3_file_to_path(s3uri)
        return load_cellpose_model_from_path(checkpoint_path)
    elif isinstance(s3uri, tuple):
        checkpoint_path = download_s3_file_to_path(s3uri[0])
        config_path = download_s3_file_to_path(s3uri[1])
        return load_cytodl_model_from_path(checkpoint_path, config_path, instantiate=instantiate)

    logger.error("Argument '%s' must be of type 'str' or 'tuple[str, str]'", s3uri)
    raise ValueError("Unable to determine model from fmsid type.")


def load_model_from_mlflow(
    mlflowid: str, *, instantiate: bool = False
) -> "CytoDLModel | BaseDiffusionAutoEncoder | DiffusionAutoEncoder":
    """
    Load model from MLFlow by run ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `mlflow` installed.

    Parameters
    ----------
    mlflowid
        MLFlow run ID.
    instantiate
        True to instantiate the model object, False otherwise.

    Returns
    -------
    :
        Model loaded from MLflow.
    """

    from endo_pipeline.io.mlflow import get_checkpoint_path_from_mlflow, get_config_path_from_mlflow

    checkpoint_path = get_checkpoint_path_from_mlflow(mlflowid)
    config_path = get_config_path_from_mlflow(mlflowid)

    return load_cytodl_model_from_path(checkpoint_path, config_path, instantiate=instantiate)


def load_cellpose_model_from_path(checkpoint_path: Path) -> "CellposeModel":
    """
    Load Cellpose model from path.

    Parameters
    ----------
    checkpoint_path
        Path to model checkpoint file.

    Returns
    -------
    :
        Cellpose model loaded from path.
    """

    from cellpose import models

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path '{checkpoint_path}' does not exist")

    return models.CellposeModel(gpu=True, pretrained_model=checkpoint_path.as_posix())


def load_cytodl_model_from_path(
    checkpoint_path: Path, config_path: Path, *, instantiate: bool = False
) -> "CytoDLModel | BaseDiffusionAutoEncoder | DiffusionAutoEncoder":
    """
    Load CytoDL model from path.

    Parameters
    ----------
    checkpoint_path
        Path to model checkpoint file
    config_path
        Path to model config file.
    instantiate
        True to instantiate the model object, False otherwise.

    Returns
    -------
    :
        CytoDL model loaded from path.
    """

    from cyto_dl.api import CytoDLModel

    model = CytoDLModel()
    model.load_config_from_file(config_path.as_posix())
    model.override_config(
        {
            "checkpoint.ckpt_path": checkpoint_path.as_posix(),
            "checkpoint.strict": True,
        }
    )

    # Instantiate model if requested.
    if instantiate:
        model = instantiate_model_target_class(model)

    return model


def load_model(
    location: ModelLocation, instantiate: bool = False
) -> "CytoDLModel | BaseDiffusionAutoEncoder | DiffusionAutoEncoder":
    """
    Load model from location with config and checkpoint, defaulting to MLFlow.

    By default, the loaded model will be a CytoDLModel.

    The specific model object can be instantiated using the model configuration,
    using the target class and weights from the associated checkpoint.

    Parameters
    ----------
    location
        Model location object.
    instantiate
        True to instantiate the model object, False otherwise.

    Returns
    -------
        Loaded model.
    """

    preferred_loader_order: list[
        tuple[str | Path | tuple[str, str] | tuple[Path, Path] | None, Callable]
    ] = [
        (location.fmsid, load_model_from_fms),
        (location.mlflowid, load_model_from_mlflow),
        (location.path, load_model_from_path),
        (location.s3uri, load_model_from_s3),
    ]

    available_loaders = [loader for loader in preferred_loader_order if loader[0] is not None]

    while available_loaders:
        field, loader = available_loaders.pop(0)
        assert field is not None

        try:
            return loader(field, instantiate=instantiate)
        except Exception as e:
            if available_loaders:
                continue
            else:
                raise e

    logger.error("Location does not have an MLFlow run ID or local path ")
    raise FileNotFoundError("Unable to load model; no available locations.")


def instantiate_model_target_class(
    model: "CytoDLModel",
) -> "BaseDiffusionAutoEncoder | DiffusionAutoEncoder":
    """
    Instantiate model target class from loaded configuration and checkpoint.

    Parameters
    ----------
    model
        Model loaded with config and checkpoint.

    Returns
    -------
    :
        Instantiated model object.
    """

    from operator import attrgetter

    from hydra.utils import get_class
    from omegaconf.errors import ConfigAttributeError

    try:
        attrgetter("model._target_", "checkpoint.ckpt_path")(model.cfg)
    except ConfigAttributeError as e:
        logger.error("Model configuration missing required key: '%s'", e.full_key)
        raise

    model_class = get_class(model.cfg.model._target_)

    if not hasattr(model_class, "load_from_checkpoint"):
        message = f"Model class [ {model_class} ] does not have a 'load_from_checkpoint' method"
        logger.error(message)
        raise ValueError(message)

    return model_class.load_from_checkpoint(model.cfg.checkpoint.ckpt_path)


def resolve_model_location(location: ModelLocation) -> str | tuple[str, str]:
    """
    Resolve model location into a POSIX path or URI, defaulting to MLFlow.

    Parameters
    ----------
    location
        Dataframe location object.

    Returns
    -------
    :
        Dataframe location as POSIX path or URI.
    """

    if location.mlflowid is not None:
        from endo_pipeline.io.mlflow import (
            get_checkpoint_path_from_mlflow,
            get_config_path_from_mlflow,
        )

        checkpoint_path = get_checkpoint_path_from_mlflow(location.mlflowid)
        config_path = get_config_path_from_mlflow(location.mlflowid)

        return (checkpoint_path.as_posix(), config_path.as_posix())

    if location.fmsid is not None:
        from endo_pipeline.io.fms import get_local_path_from_fmsid

        if isinstance(location.fmsid, str):
            return get_local_path_from_fmsid(location.fmsid).as_posix()
        elif isinstance(location.fmsid, tuple):
            return (
                get_local_path_from_fmsid(location.fmsid[0]).as_posix(),
                get_local_path_from_fmsid(location.fmsid[1]).as_posix(),
            )

        return get_local_path_from_fmsid(location.fmsid).as_posix()

    if location.path is not None:
        if isinstance(location.path, Path):
            return location.path.as_posix()
        elif isinstance(location.path, tuple):
            return (location.path[0].as_posix(), location.path[1].as_posix())

    if location.s3uri is not None:
        return location.s3uri

    logger.error("Location does not have an MLFlow ID, FMS ID, or S3 URI.")
    raise FileNotFoundError("Unable to resolve model location; no available locations.")
