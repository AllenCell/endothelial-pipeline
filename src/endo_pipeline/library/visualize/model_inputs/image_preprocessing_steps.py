import importlib
import logging
import typing
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from dask.array import Array
from matplotlib import pyplot as plt
from monai.data import MetaTensor

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig, ListConfig


logger = logging.getLogger(__name__)


def save_stack_slices_as_thumbnails(img: Array, save_dir: Path) -> None:
    """
    Save each slice of a multi-channel image stack as individual thumbnail images.

    This function iterates through all channels and slices of a Dask array representing
    an image stack, computes each slice, and saves it as a thumbnail image to the specified directory.

    Args:
        img (dask.array.Array): A Dask array representing the image stack.
            The array is expected to have the shape (channels, slices, height, width).
        save_dir (Path): The directory where the thumbnail images will be saved.
    """
    for channel, channel_name in enumerate(["bf", "cdh5"]):
        for slice_idx in range(img.shape[1]):
            slice_img = img[channel, slice_idx, :, :]
            plot_image_thumbnail(
                slice_img,
                f"{channel_name}_sliceindex{slice_idx}",
                save_dir,
                figsize=(6, 6),
                scalebar_size_um=50,
                pixel_size=PIXEL_SIZE_3i_20x,
            )


def initialize_transform(transform_cfg):
    """
    Dynamically initialize a transform object from a configuration dictionary.

    This function takes a configuration dictionary that specifies the target class
    (using the `_target_` key) and its parameters. It dynamically imports the module,
    retrieves the class, and initializes an instance of the class with the provided parameters.

    Args:
        transform_cfg (dict): A dictionary containing the configuration for the transform.
            - `_target_` (str): The full path to the class (e.g., "module.submodule.ClassName").
            - Other keys: Parameters to initialize the class.

    Returns:
        Any: An instance of the dynamically initialized class.
    """
    # Extract the module and class name from the _target_ field
    target_path = transform_cfg["_target_"]
    module_name, class_name = target_path.rsplit(".", 1)

    # Dynamically import the module and get the class
    module = importlib.import_module(module_name)
    target_class = getattr(module, class_name)

    # Extract parameters and initialize the transform
    params = {k: v for k, v in transform_cfg.items() if k != "_target_"}
    return target_class(**params)


def get_image_transforms(model_config):
    """
    Prepare and initialize the training transform pipeline with an optional squeeze operation.
    Only get the image transformation steps that process the FOV (skip load and crop steps).

    Args:
        model_config: The model configuration containing the training dataloader transform settings.

    Returns:
        list: A list of initialized transforms for the training pipeline.
    """
    # Access the training transform configuration
    train_transform_cfg = model_config.data.train_dataloaders.dataset.transform

    transforms_to_initialize = train_transform_cfg.transforms

    filtered_transforms = []
    for t in transforms_to_initialize:
        # Remove data loading step
        if t["_target_"] == "endo_pipeline.library.model.image_loading.BioIOImageLoaderd":
            continue

        # Remove RandSpatialCropSamplesd and steps after (totensor)
        if t["_target_"] == "monai.transforms.RandSpatialCropSamplesd":
            break
        filtered_transforms.append(t)

    # Initialize all remaining transforms in the pipeline
    return [initialize_transform(t) for t in filtered_transforms]


def plot_and_save_histogram(
    value_np: np.ndarray, transform_name: str, key: str, save_dir: Path, i: int
) -> None:
    """
    Plot and save a histogram of the values in a NumPy array for a given image transform.

    Args:
        value_np (np.ndarray): The NumPy array containing the values to plot.
        transform (str): The transform name whose class name is used for labeling the plot.
        key (str): channel key of the image being processed (e.g., 'raw_bf').
        save_dir (Path): The directory where the histogram plot will be saved.
        i (int): An index representing the order of the transform in the pipeline.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(value_np.ravel(), bins=50, color="grey", alpha=0.7)
    ax.set_title(f"{transform_name} ({key})", fontsize=14)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    plt.show()
    save_plot_to_path(
        fig,
        save_dir,
        f"{key}_{i}_{transform_name}_histogram",
        dpi=300,
        file_format=".pdf",
        transparent=True,
    )
    plt.close(fig)


def create_data_dict_loaded_image(
    loaded_image: np.ndarray,
) -> dict:
    """
    Create a data dictionary for BioIOImageLoaderd from an already loaded image.

    Parameters
    ----------
    loaded_image : np.ndarray
        The preloaded image data.

    Returns
    -------
    dict
        A data dictionary compatible with BioIOImageLoaderd.
    """
    # Wrap the image in a MetaTensor with metadata
    loaded_image = loaded_image.astype(np.float32)
    data = {"raw": MetaTensor(loaded_image)}
    return data


def apply_img_transforms(
    transforms: list[Any], sample: dict[str, Any]
) -> dict[str, np.ndarray | torch.Tensor]:
    """
    Apply a sequence of transforms to a sample image dictionary.
    Images in the sample dict can be NumPy arrays or torch Tensors.
    We expect all outpus to be NumPy arrays if this function is applied to the transform
    list from get_image_transforms.

    Args:
        transforms (List[Any]): Sequence of transform objects to apply.
        sample (Dict[str, Any]): Input data dictionary.

    Returns:
        Transformed sample (dict).
    """
    for i, transform in enumerate(transforms):
        logger.info("Applying Transform %s: %s", i + 1, {transform.__class__.__name__})

        sample = transform(sample)

    return sample


def get_target_image_from_sample(sample: dict[str, Any], target_key: str) -> np.ndarray:
    """
    Extract the target image from the sample dictionary. Will be numpy arrays for image transforms.

    Args:
        sample (Dict[str, Any]): Input data dictionary.
        target_key (str): Key of the target image to extract. For example, 'raw_bf', 'raw_cdh5'.

    Returns:
        np.ndarray: The extracted image as a NumPy array.
    """
    if target_key not in sample:
        logger.error("Input key '%s' not found in sample dictionary.", target_key)
        raise ValueError("Input key '%s' not found in sample dictionary.", target_key)

    value = sample[target_key]

    if isinstance(value, np.ndarray):
        return value
    elif isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    else:
        logger.error("Unsupported type for '%s': %s", target_key, type(value))
        raise TypeError("Unsupported type for '%s': %s", target_key, type(value))


def visualize_fov_transform_steps(
    transforms: list[Any],
    sample: dict[str, Any],
    save_dir: Path,
    target_key: str,
) -> np.ndarray | None:
    """
    Apply a sequence of transforms to a sample and optionally visualize.

    Args:
        transforms (List[Any]): Sequence of transforms to apply.
        sample (Dict[str, Any]): Input data dictionary.
        save_dir (Path): Directory where visualizations are saved.
        target_key (str): Key to visualize (e.g., 'raw_bf').

    Returns:
        The transformed image as a NumPy array.
    """
    for step_idx, transform in enumerate(transforms):
        transform_name = transform.__class__.__name__
        logger.info("Applying Transform %d: %s", step_idx + 1, transform_name)

        # Determine which keys this transform operates on
        transform_keys = getattr(transform, "keys", getattr(transform, "key", []))
        if isinstance(transform_keys, str):
            transform_keys = [transform_keys]

        sample = transform(sample)

        # Visualize if target_key affected
        if target_key in transform_keys and isinstance(sample, dict):
            value_np = get_target_image_from_sample(sample, target_key)

            plot_image_thumbnail(
                value_np.squeeze(),
                f"{target_key}_{step_idx + 1}_{transform_name}",
                save_dir,
                figsize=(6, 6),
                scalebar_size_um=50,
                pixel_size=PIXEL_SIZE_3i_20x,
            )
            plot_and_save_histogram(
                value_np.squeeze(), transform_name, target_key, save_dir, step_idx
            )

    transformed_image = get_target_image_from_sample(sample, target_key)
    return transformed_image
