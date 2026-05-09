"""Methods for loading models."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cyto_dl.api import CytoDLModel
    from cyto_dl.models.im2im.diffusion_autoencoder import (
        DiffusionAutoEncoder as BaseDiffusionAutoEncoder,
    )

    from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder

from endo_pipeline.manifests import ModelLocation

logger = logging.getLogger(__name__)


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

    from endo_pipeline.io.mlflow import get_checkpoint_path_from_mlflow, get_config_dict_from_mlflow

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
