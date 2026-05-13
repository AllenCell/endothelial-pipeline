"""Methods for loading inputs."""

import logging
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from cyto_dl.api import CytoDLModel
    from cyto_dl.models.im2im.diffusion_autoencoder import (
        DiffusionAutoEncoder as BaseDiffusionAutoEncoder,
    )
    from omegaconf import DictConfig, ListConfig

    from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder


from endo_pipeline.io.output import get_output_path
from endo_pipeline.manifests import ModelLocation

logger = logging.getLogger(__name__)


def get_repository_root_dir() -> Path:
    """
    Get path to root of git repository.

    Returns
    -------
    :
        Path object for root of git repository.
    """

    return Path(__file__).resolve().parents[3]


def get_config_dict_from_mlflow(mlflowid: str) -> "DictConfig | ListConfig":
    """
    Get config dict from given MLFlow run ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `mlflow` installed.

    Parameters
    ----------
    mlflowid
        MLFlow run ID.

    Returns
    -------
    :
        Loaded config dict.
    """

    from omegaconf import OmegaConf

    from endo_pipeline.io.mlflow import MLFLOW

    # Check if config artifact exists
    configs = MLFLOW.artifacts.list_artifacts(run_id=mlflowid, artifact_path="config")

    # If no config artifacts are found, we cannot load the model
    if len(configs) == 0:
        logger.error("No config artifacts found for run id [ %s ]", mlflowid)
        raise LookupError("No config artifacts found")

    # If multiple config artifacts are found, default to using the first in the
    # list, but log a warning for the user
    if len(configs) > 1:
        logger.warning("Multiple config artifacts found for run id [ %s ]", mlflowid)

    logger.debug("Loading model config [ %s ]", configs[0].path)

    # Define config URI for loading the artifact
    config_uri = f"runs:/{mlflowid}/{configs[0].path}"
    return OmegaConf.create(MLFLOW.artifacts.load_text(config_uri))


def get_checkpoint_path_from_mlflow(mlflowid: str) -> Path:
    """
    Get local path to checkpoint file from given MLFlow run ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `mlflow` installed.

    Parameters
    ----------
    mlflowid
        MLFlow run ID.

    Returns
    -------
    :
        Local path to checkpoint file. If both "last.ckpt" and "best.ckpt"
        are available, defaults to "last.ckpt".
    """

    from endo_pipeline.io.mlflow import MLFLOW

    # Check if checkpoint is already downloaded.
    path = get_output_path("model_checkpoints", mlflowid, include_timestamp=False)
    last_checkpoint_path = path / "last.ckpt"
    best_checkpoint_path = path / "best.ckpt"

    if last_checkpoint_path.exists():
        logger.warning(
            "Last checkpoint for run [ %s ] available at [ %s ]. "
            "Using this checkpoint. If you want to redownload the artifact, delete this file.",
            mlflowid,
            last_checkpoint_path,
        )
        return last_checkpoint_path

    if best_checkpoint_path.exists():
        logger.warning(
            "Best checkpoint for run [ %s ] available at [ %s ]. "
            "Using this checkpoint. If you want to redownload the artifact, delete this file.",
            mlflowid,
            best_checkpoint_path,
        )
        return best_checkpoint_path

    # Find all available checkpoints
    artifacts = MLFLOW.artifacts.list_artifacts(run_id=mlflowid, artifact_path="checkpoints")
    directories = [artifact for artifact in artifacts if artifact.is_dir]
    checkpoints = [artifact.path for artifact in artifacts if not artifact.is_dir]

    # Continue iterating through artifacts if there are directories
    while directories:
        artifact = directories.pop()
        artifacts = MLFLOW.artifacts.list_artifacts(run_id=mlflowid, artifact_path=artifact.path)
        directories.extend([artifact for artifact in artifacts if artifact.is_dir])
        checkpoints.extend([artifact.path for artifact in artifacts if not artifact.is_dir])

    # Filter artifacts for "last.ckpt" and "best.ckpt"
    last_checkpoint = [ckpt for ckpt in checkpoints if ckpt.endswith("last.ckpt")]
    best_checkpoint = [ckpt for ckpt in checkpoints if ckpt.endswith("best.ckpt")]

    # If neither option is found, throw an error
    if not last_checkpoint and not best_checkpoint:
        logger.error("No valid checkpoint artifacts found for run id [ %s ]", mlflowid)
        raise LookupError("No checkpoint artifacts found")

    # Build checkpoint artifact URI
    checkpoint = last_checkpoint[0] if last_checkpoint else best_checkpoint[0]
    checkpoint_uri = f"runs:/{mlflowid}/{checkpoint}"

    # Download artifact to output location and return path
    return Path(
        MLFLOW.artifacts.download_artifacts(artifact_uri=checkpoint_uri, dst_path=path.as_posix())
    )


def load_model_from_mlflow(
    mlflowid: str, instantiate: bool = False
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

    from cyto_dl.api import CytoDLModel

    config_dict = get_config_dict_from_mlflow(mlflowid)
    checkpoint_path = get_checkpoint_path_from_mlflow(mlflowid)

    model = CytoDLModel()
    model.load_config_from_dict(config_dict)
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

    if location.mlflowid is not None:
        return load_model_from_mlflow(location.mlflowid, instantiate)

    logger.error("Location does not have an MLFlow run ID.")
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
