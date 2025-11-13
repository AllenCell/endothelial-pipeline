import logging
import typing

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


def get_latent_dim_from_config(model_config: "DictConfig | ListConfig") -> int:
    """Get latent dimension size from model config."""
    from omegaconf import OmegaConf

    return OmegaConf.select(model_config, "model.semantic_encoder.base_encoder.num_classes")
