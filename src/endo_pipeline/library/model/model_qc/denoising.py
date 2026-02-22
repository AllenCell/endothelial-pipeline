"""Denoising experiment helpers for model QC."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
from endo_pipeline.library.model.diffae.generate_image import (
    add_noise_to_image,
    generate_from_coords_and_noised_image,
)

if TYPE_CHECKING:
    from numpy.random import Generator

    from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder

logger = logging.getLogger(__name__)


def run_denoising_experiments(
    model: DiffusionAutoEncoder,
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    rng: Generator,
    noise_levels: list[float],
    num_gpus: int,
) -> dict[str, list[np.ndarray]]:
    """Run denoising experiments with normal and negative control conditioning.

    Three experiments are run:

    1. **Normal conditioning** — denoise using the true conditioning latent.
    2. **Scrambled embedding** — denoise using a randomly permuted latent vector.
    3. **Scrambled input** — denoise using a latent extracted from a
       pixel-scrambled version of the conditioning image.

    Parameters
    ----------
    model
        The loaded diffusion autoencoder model.
    conditioning_input_crop
        Cropped conditioning input image.
    diffusion_input_crop
        Cropped diffusion input (ground truth) image.
    rng
        NumPy random number generator for reproducibility.
    noise_levels
        Fractional noise levels to apply (e.g. ``[0.25, 0.5, 0.75]``).
    num_gpus
        Number of GPUs available for inference.

    Returns
    -------
    results
        Dictionary with keys:

        - ``"images_to_denoise"`` — noised images at each level plus pure noise.
        - ``"noise_image"`` — the random noise image (single-element list).
        - ``"denoised_normal"`` — denoised using the true conditioning latent.
        - ``"denoised_scrambled_embedding"`` — denoised using a randomly permuted latent vector.
        - ``"denoised_scrambled_input"`` — denoised using a latent from a pixel-scrambled conditioning image.
        - ``"conditioning_latent"`` — the original latent vector (single-element list).
    """

    latent = get_latent_vector_from_crop(model, conditioning_input_crop, num_gpus=num_gpus)
    noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

    noisy_images = [
        add_noise_to_image(diffusion_input_crop, noise_image, level) for level in noise_levels
    ]
    images_to_denoise = [*noisy_images, noise_image]

    # Normal conditioning
    denoised_normal = [
        generate_from_coords_and_noised_image(model, latent, img, num_gpus=num_gpus)
        for img in images_to_denoise
    ]

    # Scrambled latent vector
    latent_scrambled = rng.permuted(latent)
    denoised_scrambled_emb = [
        generate_from_coords_and_noised_image(model, latent_scrambled, img, num_gpus=num_gpus)
        for img in images_to_denoise
    ]

    # Latent from scrambled input image
    img_scrambled = rng.permuted(conditioning_input_crop.ravel()).reshape(
        conditioning_input_crop.shape
    )
    latent_from_scrambled = get_latent_vector_from_crop(model, img_scrambled, num_gpus=num_gpus)
    denoised_scrambled_input = [
        generate_from_coords_and_noised_image(model, latent_from_scrambled, img, num_gpus=num_gpus)
        for img in images_to_denoise
    ]

    return {
        "images_to_denoise": images_to_denoise,
        "noise_image": [noise_image],
        "denoised_normal": denoised_normal,
        "denoised_scrambled_embedding": denoised_scrambled_emb,
        "denoised_scrambled_input": denoised_scrambled_input,
        "conditioning_latent": [latent],
    }
