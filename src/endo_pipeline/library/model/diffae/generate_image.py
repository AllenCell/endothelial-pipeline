import logging
import typing

import numpy as np
import torch
from hydra.utils import get_class

if typing.TYPE_CHECKING:
    from cyto_dl.api import CytoDLModel

from endo_pipeline.configs import instantiate_diffusion_autoencoder_object

logger = logging.getLogger(__name__)


def generate_from_coords(
    model: "CytoDLModel",
    coords: np.ndarray | list[list[float]],
    n_noise_samples: int = 1,
    average: bool = False,
    num_gpus: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic image from a list of coordinates
    in the latent space of a model.

    Parameters
    ----------
    model
        The model to use for generation.
    coords
        A list of coordinates in the latent space of the model.
    n_noise_samples
        The number of noise samples to use for generation.
    average
        Whether to average the generated images.
    num_gpus
        Optional, number of available GPUs.
    """
    if not isinstance(coords, np.ndarray):
        if isinstance(coords, list):
            coords_np = np.array(coords)
        else:
            logger.error("Parameter `coords` must be a numpy array or a list of lists.")
            raise ValueError("coords must be a numpy array or a list of lists")
    else:
        coords_np = coords

    coords_torch = torch.from_numpy(coords_np).float()

    # have to instantiate the actual model object from the config
    model_instantiated = instantiate_diffusion_autoencoder_object(model.cfg)

    # move model and inputs to gpu if available, else
    # perform reconstruction on cpu
    if num_gpus:
        coords_ = coords_torch.to("cuda")
        model_ = model_instantiated.to("cuda")
    else:
        coords_ = coords_torch
        model_ = model_instantiated

    walk_img = model_.generate_from_latent(
        coords_, n_noise_samples=n_noise_samples, average=average, save=False
    )
    return walk_img


def generate_from_coords_batch(
    model: "CytoDLModel",
    coords_batch: np.ndarray | list[list[list[float]]],
    num_gpus: int | None = None,
) -> list[np.ndarray]:
    """
    Generate synthetic images from a batch of coordinates
    in the latent space of a model.

    Parameters
    ----------
    model:
        The model to use for generation.
    coords_batch:
        A batch of lists of coordinates in the latent space of the model.
    num_gpus:
        Optional, number of available GPUs.
    """

    # note to self: need to debug what the input type is here
    # I think the outlier is the latent walk? or maybe the crop
    # reconstruction? need to check
    if isinstance(coords_batch, np.ndarray):
        coords_concat = coords_batch.copy()
    elif isinstance(coords_batch, list):
        coords_concat = np.array(coords_batch)
    else:
        coords_concat = np.concatenate(coords_batch, axis=0)
    logger.debug("Concatenated coordinates shape: [ %s ]", coords_concat.shape)

    img = generate_from_coords(model, coords_concat, num_gpus=num_gpus)
    walk_imgs = [img[i] for i in range(len(coords_batch))]

    return walk_imgs
