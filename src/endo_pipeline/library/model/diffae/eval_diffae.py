import logging
import typing

import numpy as np
import torch

if typing.TYPE_CHECKING:
    from cyto_dl.models.im2im.diffusion_autoencoder import DiffusionAutoEncoder as _BaseDiffAE

    from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder

logger = logging.getLogger(__name__)


def get_latent_vector_from_crop(
    model: "DiffusionAutoEncoder | _BaseDiffAE", image_crop: np.ndarray, num_gpus: int | None = None
) -> np.ndarray:
    """
    Get the latent vector from an image using the model's semantic encoder.

    **Method inputs**

    The input image can have shape ``(num_px_y, num_px_x)``, ``(1, num_px_y, num_px_x)``,
    or ``(1, 1, num_px_y, num_px_x)``. If the input image has 2 or 3 dimensions,
    the necessary batch and channel dimensions will be added automatically (and set to 1).

    Parameters
    ----------
    model
        The diffusion autoencoder model with a semantic encoder.
    image_crop
        The image to encode.
    num_gpus
        The number of GPUs available for computation. If ``None``, defaults to CPU.
    """
    # check that input image_crop has correct number of dims
    if len(image_crop.shape) == 2:
        # Expand dims to (1, 1, num_px_y, num_px_x)
        # i.e., add batch and channel dims
        logger.debug(
            "Expanding image_crop shape from [ %s ] to [ %s ]",
            image_crop.shape,
            np.expand_dims(image_crop, axis=(0, 1)).shape,
        )
        image_crop = np.expand_dims(image_crop, axis=(0, 1))
    elif len(image_crop.shape) == 3:
        # Expand dims to (1, 1, num_px_y, num_px_x)
        # i.e., add batch dim
        logger.debug(
            "Expanding image_crop shape from [ %s ] to [ %s ]",
            image_crop.shape,
            np.expand_dims(image_crop, axis=0).shape,
        )
        image_crop = np.expand_dims(image_crop, axis=0)
    elif len(image_crop.shape) != 4:
        raise ValueError(
            f"Input image_crop must have 2, 3, or 4 dimensions, but got shape {image_crop.shape}"
        )

    # check that model has semantic_encoder attribute
    if not hasattr(model, "semantic_encoder"):
        raise AttributeError("The provided model does not have a 'semantic_encoder' attribute.")

    # convert to torch tensor
    image_crop_torch = torch.from_numpy(image_crop).float()

    # move model and inputs to gpu if available, else
    # perform reconstruction on cpu
    if num_gpus:
        image_crop_ = image_crop_torch.to("cuda")
        model_ = model.to("cuda")
    else:
        image_crop_ = image_crop_torch
        model_ = model

    with torch.no_grad():
        latent_vector: torch.Tensor = model_.semantic_encoder(image_crop_)

    # Move latent vector back to cpu and convert to numpy
    latent_vector_np: np.ndarray = latent_vector.cpu().numpy()
    # Squeeze to remove batch dimension
    latent_vector_np = np.squeeze(latent_vector_np, axis=0)

    return latent_vector_np
