import logging
import typing

import numpy as np
import torch
from hydra.utils import get_class

if typing.TYPE_CHECKING:
    from cyto_dl.api import CytoDLModel

from endo_pipeline.configs import instantiate_diffusion_autoencoder_object

logger = logging.getLogger(__name__)


def add_noise_to_image(
    input_image: np.ndarray,
    noise_image: np.ndarray,
    noise_level: float,
    clip: bool = False,
    random_seed: int = 47,
) -> np.ndarray:
    """
    Add Gaussian noise to an input image at a specified noise level.

    **Noise level weighting**

    The output "noised" image is created using the formula:

        output_image = (1 - noise_level) * input_image + (noise_level) * noise_img

    Using this formula, ``noise_level`` represents the fraction of the corrupted image
    that is contributed by the noise image, with the remainder contributed by the original input image.
    An input noise_level of 0.0 results in no noise being added (the output image is identical
    to the input image), while a noise_level of 1.0 results in an image composed entirely of noise.

    Parameters
    ----------
    input_image
        The input image to which noise will be added.
    noise_image
        A standard Gaussian noise image of the same shape as the input image.
    noise_level
        The level of noise to add, between 0.0 (no noise) and 1.0 (all noise).
    clip
        Whether to clip the output image to the valid range [0, 1].
    random_seed
        Seed for the random number generator for reproducibility.
    """
    if not (0.0 <= noise_level <= 1.0):
        logger.error("Parameter `noise_level` must be between 0.0 and 1.0.")
        raise ValueError("Parameter `noise_level` must be between 0.0 and 1.0.")

    # Check edge cases for numerical efficiency
    if noise_level == 0.0:
        output_image = input_image.copy()
    elif noise_level == 1.0:
        output_image = noise_image.copy()
    else:  # general case
        output_image = np.sqrt(1 - noise_level) * input_image + np.sqrt(noise_level) * noise_image

    # Clip the output image to the valid range [0, 1] if specified
    if clip:
        output_image = np.clip(output_image, 0.0, 1.0)
    return output_image


def generate_from_coords_and_noised_image(
    model: "CytoDLModel",
    coords: np.ndarray,
    noised_image: np.ndarray,
    num_gpus: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic image by denoising a noised image
    conditioned on a coordinate in the latent space of a model.

    **Input array shapes**

    The input conditioning vector array should have shape ``(num_vecs, num_dims)``, where
    ``num_vecs`` is the number of conditioning vectors and ``num_dims`` is the dimensionality
    of the latent space. This allows for generating multiple images corresponding
    to ``num_vecs`` different conditioning vectors.

    The noised image tensor should have shape ``(num_channels, num_pixels_y, num_pixels_x)``,
    where ``num_channels`` is the number of channels, ``num_pixels_y`` is the height of the image
    (number of pixels in Y), and ``num_pixels_x`` is the width of the image (number of pixels in X).
    Note that this shape should be the same as ``model.image_shape`` in the model's configuration.

    **Example usage**

    .. code-block:: python

        from endo_pipeline.io import load_model
        from endo_pipeline.manifests import load_model_manifest
        from endo_pipeline.library.model.diffae import generate_from_coords_and_noised_image

        model_manifest = load_model_manifest("my_model_manifest")
        model_location = model_manifest.locations["my_run_name"]
        model = load_model(model_location)

        gen_image = generate_from_coords_and_noised_image(
            model,
            coords=my_coords, # shape (num_vecs, num_dims)
            noised_image=my_noised_image, # shape (1, n_y, n_x) for this model
        )

    Parameters
    ----------
    model
        The model to use for image generation (conditioned denoising).
    coords
        A coordinate in the latent space of the model; used to condition the denoising.
    noised_image
        An image used as the starting point for denoising by the model.
    num_gpus
        Optional, number of available GPUs.
    """
    coords_torch = torch.from_numpy(coords).float()
    noised_image_torch = torch.from_numpy(noised_image).float()

    # have to instantiate the actual model object from the config
    model_class = get_class(model.cfg.model._target_)
    model_instantiated = model_class.load_from_checkpoint(model.cfg.checkpoint.ckpt_path)  # type: ignore[attr-defined]

    # move model and inputs to gpu if available, else
    # perform reconstruction on cpu
    if num_gpus:
        coords_ = coords_torch.to("cuda")
        noised_image_ = noised_image_torch.to("cuda")
        model_ = model_instantiated.to("cuda")
    else:
        coords_ = coords_torch
        noised_image_ = noised_image_torch
        model_ = model_instantiated

    if not hasattr(model_, "generate_from_latent_and_noised_image"):
        logger.error(
            "Model class [ %s ] does not support generation from coordinates and noised image.",
            model_.__class__.__name__,
        )
        raise NotImplementedError(
            f"Model class [ {model_.__class__.__name__} ] does not support generation from coordinates and noised image."
        )

    gen_img = model_.generate_from_latent_and_noised_image(
        conditioning_vector=coords_,
        noised_image=noised_image_,
    )
    return gen_img


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
