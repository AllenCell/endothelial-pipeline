import logging
import typing

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


def get_latent_dim_from_config(model_config: "DictConfig | ListConfig") -> int:
    """Get latent dimension size from model config."""
    if not hasattr(model_config.model, "semantic_encoder"):
        logger.error("Model config does not have 'semantic_encoder' attribute.")
        raise AttributeError("Model config does not have 'semantic_encoder' attribute.")
    if not hasattr(model_config.model.semantic_encoder, "base_encoder"):
        logger.error("Model config does not have 'base_encoder' attribute in 'semantic_encoder'.")
        raise AttributeError(
            "Model config does not have 'base_encoder' attribute in 'semantic_encoder'."
        )
    if not hasattr(model_config.model.semantic_encoder.base_encoder, "num_classes"):
        logger.error("Model config does not have 'num_classes' attribute in 'base_encoder'.")
        raise AttributeError(
            "Model config does not have 'num_classes' attribute in 'base_encoder'."
        )

    return model_config.model.semantic_encoder.base_encoder.num_classes
