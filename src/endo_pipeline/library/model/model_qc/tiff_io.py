"""TIFF file I/O helpers for saving model QC crops."""

import logging
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def save_image_as_tiff(image: np.ndarray, output_path: Path, filename: str) -> None:
    """Save an image array as a TIFF file.

    Parameters
    ----------
    image
        Image array to save. Will be squeezed and cast to float32 if needed.
    output_path
        Directory where the TIFF file will be written.
    filename
        Base name for the output file (without extension).
    """
    output_file = output_path / f"{filename}.tiff"
    image_to_save = image.squeeze()
    if not np.issubdtype(image_to_save.dtype, np.floating):
        image_to_save = image_to_save.astype(np.float32)
    tifffile.imwrite(output_file, image_to_save)


def save_denoising_crops(
    output_path: Path,
    dataset_name: str,
    position: int,
    timepoint: int,
    start_x: int,
    start_y: int,
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    noisy_diffusion_input_images: list[np.ndarray],
    noise_image: np.ndarray,
    denoised_images: list[np.ndarray],
    noise_levels: list[float],
) -> Path:
    """Save all denoising crops as TIFF files to a structured directory.

    Parameters
    ----------
    output_path
        Root output directory.
    dataset_name
        Name of the dataset.
    position
        Field-of-view position index.
    timepoint
        Timepoint index.
    start_x
        Crop X start coordinate.
    start_y
        Crop Y start coordinate.
    conditioning_input_crop
        Cropped conditioning input image.
    diffusion_input_crop
        Cropped ground-truth diffusion input image.
    noisy_diffusion_input_images
        Noised versions of the diffusion input at each noise level.
    noise_image
        The pure noise image.
    denoised_images
        Denoised outputs at each noise level.
    noise_levels
        Fractional noise levels corresponding to ``noisy_diffusion_input_images``.

    Returns
    -------
    crops_output_path
        The directory where crop files were saved.
    """
    crops_output_path = (
        output_path / "crops" / f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}"
    )
    crops_output_path.mkdir(parents=True, exist_ok=True)

    save_image_as_tiff(conditioning_input_crop, crops_output_path, "conditioning_input")
    save_image_as_tiff(diffusion_input_crop, crops_output_path, "ground_truth")

    for noised_img, noise_level in zip(noisy_diffusion_input_images, noise_levels, strict=False):
        save_image_as_tiff(noised_img, crops_output_path, f"noised_{int(noise_level * 100):03d}pct")

    save_image_as_tiff(noise_image, crops_output_path, "noised_100pct")

    # `denoised_images` is either (a) one image per fractional noise level
    # plus a trailing 100 % entry, or (b) a single 100 % entry (metrics-only
    # mode).  Key the pct label off whether we're at the last position, not
    # the index, so a lone 100 % denoising isn't mislabelled as the first
    # fractional level.
    num_denoised = len(denoised_images)
    for idx, denoised_img in enumerate(denoised_images):
        is_last = idx == num_denoised - 1
        pct = 100 if is_last else int(noise_levels[idx] * 100)
        save_image_as_tiff(denoised_img, crops_output_path, f"denoised_from_{pct:03d}pct_noise")

    logger.debug("Saved crops to %s", crops_output_path)
    return crops_output_path
