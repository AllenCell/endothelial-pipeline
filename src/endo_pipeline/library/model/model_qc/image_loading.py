"""Image loading and preprocessing helpers for model QC."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    apply_img_transforms,
    create_data_dict_loaded_image,
    get_image_transforms,
    get_target_image_from_sample,
)
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from endo_pipeline.settings.examples import ExampleImage

logger = logging.getLogger(__name__)


def load_transformed_image(
    example: "ExampleImage",
    model_config: "DictConfig",
    timepoint: int | None = None,
) -> dict[str, Any]:
    """Load a zarr image, apply transforms, and return the transformed sample.

    Parameters
    ----------
    example
        The example image metadata (dataset, position, timepoint, etc.).
    model_config
        The model configuration loaded from MLflow.
    timepoint
        Timepoint to load. If None, uses ``example.timepoint``.

    Returns
    -------
    sample
        The transformed image data dictionary.
    """
    dataset_config = load_dataset_config(example.dataset_name)
    zarr_loc = get_zarr_location_for_position(dataset_config, example.position)
    tp = timepoint if timepoint is not None else example.timepoint
    img = load_image(
        zarr_loc,
        level=DIFFAE_ZARR_RESOLUTION_LEVEL,
        timepoints=tp,
        squeeze=True,
        compute=True,
    )
    data = create_data_dict_loaded_image(img)
    transforms = get_image_transforms(model_config)
    sample = apply_img_transforms(transforms, data)
    return sample


def load_and_preprocess_example_crop(
    example: "ExampleImage",
    model_config: "DictConfig",
    crop_size: int,
    channel_key_for_conditioning_input: str,
    diffusion_input_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image for a given example and preprocess it for model QC.

    Parameters
    ----------
    example
        The example image metadata.
    model_config
        The model configuration loaded from MLflow.
    crop_size
        Size of the square crop in pixels.
    channel_key_for_conditioning_input
        Key for the conditioning input channel (e.g. ``"raw_bf"``).
    diffusion_input_key
        Key for the diffusion input channel.

    Returns
    -------
    conditioning_crop
        Cropped conditioning input image.
    diffusion_crop
        Cropped diffusion input image.
    """
    sample = load_transformed_image(example, model_config)

    conditioning_img = get_target_image_from_sample(
        sample, target_key=channel_key_for_conditioning_input
    )
    diffusion_img = get_target_image_from_sample(sample, target_key=diffusion_input_key)

    conditioning_crop = crop_image(
        conditioning_img,
        example.crop_x_start,
        example.crop_y_start,
        crop_size,
    )
    diffusion_crop = crop_image(
        diffusion_img,
        example.crop_x_start,
        example.crop_y_start,
        crop_size,
    )
    return conditioning_crop, diffusion_crop
