import logging
import typing

from hydra.utils import get_class

if typing.TYPE_CHECKING:
    from cyto_dl.models.im2im.diffusion_autoencoder import DiffusionAutoEncoder as _BaseDiffAE
    from omegaconf import DictConfig, ListConfig

    from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder

logger = logging.getLogger(__name__)


def instantiate_diffusion_autoencoder_object(
    model_config: "DictConfig | ListConfig",
) -> "_BaseDiffAE | DiffusionAutoEncoder":
    """
    Instantiate a DiffusionAutoEncoder object from a model configuration.

    The model configuration must contain the necessary information to
    instantiate the model, including the target class and checkpoint path.

    The returned object will be an instance of the model class specified
    in the config initialized with the weights from the provided checkpoint.

    Parameters
    ----------
    model_config
        The model configuration dictionary.

    Returns
    -------
    :
        An instantiated Diffusion Autoencoder object.
    """
    # Validate model_config structure
    # Must have 'model' and 'checkpoint' keys with required sub-keys
    # in order to instantiate the model correctly.
    if not hasattr(model_config, "model"):
        logger.error("Model configuration must have a 'model' key.")
        raise ValueError("Model configuration must have a 'model' key.")
    if not hasattr(model_config.model, "_target_"):
        logger.error("Model configuration 'model' must have a '_target_' key.")
        raise ValueError("Model configuration 'model' must have a '_target_' key.")
    if not hasattr(model_config, "checkpoint"):
        logger.error("Model configuration must have a 'checkpoint' key.")
        raise ValueError("Model configuration must have a 'checkpoint' key.")
    if not hasattr(model_config.checkpoint, "ckpt_path"):
        logger.error("Model configuration 'checkpoint' must have a 'ckpt_path' key.")
        raise ValueError("Model configuration 'checkpoint' must have a 'ckpt_path' key.")

    model_class = get_class(model_config.model._target_)

    # Instantiate the model from the checkpoint, expecting the class to have
    # to have a `load_from_checkpoint` method.
    if not hasattr(model_class, "load_from_checkpoint"):
        logger.error(
            f"The model class [ {model_class} ] does not have a 'load_from_checkpoint' method."
        )
        raise ValueError(
            f"The model class [ {model_class} ] does not have a 'load_from_checkpoint' method."
        )

    model_instantiated = model_class.load_from_checkpoint(model_config.checkpoint.ckpt_path)

    return model_instantiated
