"""Methods for assessing model performance."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from bioio_base.types import PhysicalPixelSizes
from numpy.random import Generator

from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
from endo_pipeline.library.model.diffae.generate_image import generate_from_coords_and_noised_image
from endo_pipeline.library.process.general_image_preprocessing import save_image_output
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1 as PIXEL_SIZE

if TYPE_CHECKING:
    from cyto_dl.models.im2im.diffusion_autoencoder import (
        DiffusionAutoEncoder as BaseDiffusionAutoEncoder,
    )
    from numpy.typing import NDArray

    from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder


logger = logging.getLogger(__name__)


def add_noise_to_image(
    input_image: np.ndarray,
    noise_image: np.ndarray,
    noise_level: float,
) -> np.ndarray:
    """
    Add Gaussian noise to an input image at a specified noise level.

    The output "noised" image is created using the formula:

    output_image = sqrt(1 - noise_level) * input_image + sqrt(noise_level) * noise_image

    Using this formula, `noise_level` represents the fraction of the corrupted
    image that is contributed by the noise image, with the remainder contributed
    by the original input image. An input `noise_level` of `0.0` results in no
    noise being added (the output image is identical to the input image), while
    a `noise_level` of `1.0` results in an image composed entirely of noise.

    Parameters
    ----------
    input_image
        The input image to which noise will be added.
    noise_image
        A standard Gaussian noise image of the same shape as the input image.
    noise_level
        The level of noise to add, between 0.0 (no noise) and 1.0 (all noise).

    Returns
    -------
    :
        The resulting noised image.
    """

    if not (0.0 <= noise_level <= 1.0):
        logger.error("Parameter 'noise_level' must be between 0.0 and 1.0.")
        raise ValueError("Parameter 'noise_level' must be between 0.0 and 1.0.")

    if noise_level == 0.0:
        output_image = input_image.copy()
    elif noise_level == 1.0:
        output_image = noise_image.copy()
    else:
        output_image = np.sqrt(1 - noise_level) * input_image + np.sqrt(noise_level) * noise_image

    return output_image


def denoise_with_scrambled_latent_vector(
    rng: Generator,
    model: "BaseDiffusionAutoEncoder | DiffusionAutoEncoder",
    noise_examples: list["NDArray"],
    latent: "NDArray",
    num_gpus: int | None = None,
) -> list["NDArray"]:
    """
    Scramble latent vector before denoising the given noise examples.

    Parameters
    ----------
    rng
        Random number generator instance.
    model
        Instantiated model.
    noise_examples
        Noised image examples.
    latent
        Latent vector from conditioning image example.
    num_gpus
        Number of GPUs to use. If None, run on CPU.
    """

    scrambled_latent = rng.permuted(latent)
    return [
        generate_from_coords_and_noised_image(model, scrambled_latent, crop, num_gpus)
        for crop in noise_examples
    ]


def denoise_with_scrambled_conditioning_input(
    rng: Generator,
    model: "BaseDiffusionAutoEncoder | DiffusionAutoEncoder",
    noised_examples: list["NDArray"],
    conditioning_example: "NDArray",
    num_gpus: int | None = None,
) -> list["NDArray"]:
    """
    Scramble the conditioning input before denoising the given noise examples.

    Parameters
    ----------
    rng
        Random number generator instance.
    model
        Instantiated model.
    noised_examples
        Noised image examples.
    conditioning_example
        Conditioning image example.
    num_gpus
        Number of GPUs to use. If None, run on CPU.
    """

    scrambled_conditioning = rng.permuted(conditioning_example.ravel()).reshape(
        conditioning_example.shape
    )
    latent_scrambled_input = get_latent_vector_from_crop(
        model,
        scrambled_conditioning,
        num_gpus=num_gpus,
    )

    return [
        generate_from_coords_and_noised_image(model, latent_scrambled_input, crop, num_gpus)
        for crop in noised_examples
    ]


def save_model_performance_conditioning_and_diffusion_examples(
    output_path: Path,
    example: ExampleImage,
    conditioning_example: "NDArray",
    diffusion_example: "NDArray",
    dtype: type = np.float32,
) -> None:
    """
    Save conditioning and diffusions examples to tiff.

    Parameters
    ----------
    output_path
        Output path for examples.
    example
        Example image defining specific crop example.
    conditioning_example
        Conditioning image example.
    diffusion_example
        Diffusion image example.
    dtype
        Data type for output image.
    """

    base_metadata = {
        "image_name": str(example),
        "channel_colors": [(255, 255, 255)],
        "physical_pixel_sizes": PhysicalPixelSizes(PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE),
        "dim_order": "YX",
    }

    conditioning_metadata = base_metadata.copy()
    conditioning_metadata["channel_names"] = ["Conditioning"]

    diffusion_metadata = base_metadata.copy()
    diffusion_metadata["channel_names"] = ["Diffusion"]

    conditioning_path = output_path / f"{example}_conditioning_example.ome.tiff"
    diffusion_path = output_path / f"{example}_diffusion_example.ome.tiff"

    save_image_output(
        conditioning_path, [conditioning_example.squeeze()], conditioning_metadata, dtype
    )
    save_image_output(diffusion_path, [diffusion_example.squeeze()], diffusion_metadata, dtype)


def save_model_performance_denoising_examples(
    output_path: Path,
    example: ExampleImage,
    noise_examples: list,
    denoised_examples: list,
    noise_levels: list[float],
    dtype: type = np.float32,
):
    """
    Save noise and denoised examples to tiff.

    All noised examples are combined into a single tiff, with different noise
    levels in each channel. All denoised examples are combined into a single
    tiff, with different noise levels in each channel.

    Parameters
    ----------
    output_path
        Output path for examples.
    example
        Example image defining specific crop example.
    noise_examples
        Noised image examples.
    denoised_examples
        Denoised image examples.
    noise_levels
        Noise levels applied to each example.
    dtype
        Data type for output image.
    """

    metadata = {
        "image_name": str(example),
        "channel_names": [f"{level * 100:.0f}% Noise" for level in noise_levels],
        "channel_colors": [(255, 255, 255)] * len(noise_levels),
        "physical_pixel_sizes": PhysicalPixelSizes(PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE),
        "dim_order": "YX",
    }

    noised_path = output_path / f"{example}_noised_examples.ome.tiff"
    denoised_path = output_path / f"{example}_denoised_examples.ome.tiff"

    save_image_output(noised_path, [crop.squeeze() for crop in noise_examples], metadata, dtype)
    save_image_output(
        denoised_path, [crop.squeeze() for crop in denoised_examples], metadata, dtype
    )
