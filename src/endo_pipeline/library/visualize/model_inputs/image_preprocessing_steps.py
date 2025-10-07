import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dask.array import Array
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from endo_pipeline.io.output import save_plot_to_path, save_thumbnail_to_path
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

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
    for channel in range(img.shape[0]):
        for slice_idx in range(img.shape[1]):
            slice_img = img[channel, slice_idx, :, :].compute()
            save_thumbnail_to_path(
                slice_img,
                f"original_channel{channel}_slice{slice_idx}",
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


def prepare_train_transforms(
    model_config,
):
    """
    Prepare and initialize the training transform pipeline with an optional squeeze operation.

    Args:
        model_config: The model configuration containing the training dataloader transform settings.

    Returns:
        list: A list of initialized transforms for the training pipeline.
    """
    # Access the training transform configuration
    train_transform_cfg = model_config.data.train_dataloaders.dataset.transform

    # Define the squeeze transform
    temp_squeeze = {"_target_": "monai.transforms.SqueezeDimd", "keys": "raw", "dim": 0}

    # Insert the squeeze transform into the pipeline
    train_transform_cfg.transforms.insert(1, OmegaConf.create(temp_squeeze))

    # Initialize all transforms in the pipeline
    return [initialize_transform(t) for t in train_transform_cfg.transforms]


def plot_and_save_histogram(
    value_np: np.ndarray, transform: Any, key: str, save_dir: Path, i: int
) -> None:
    """
    Plot and save a histogram of the values in a NumPy array for a given image transform.

    Args:
        value_np (np.ndarray): The NumPy array containing the values to plot.
        transform (Any): The transform object whose class name is used for labeling the plot.
        key (str): channel key of the image being processed (e.g., 'raw_bf').
        save_dir (Path): The directory where the histogram plot will be saved.
        i (int): An index representing the order of the transform in the pipeline.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(value_np.ravel(), bins=50, color="grey", alpha=0.7)
    ax.set_title(f"{transform.__class__.__name__} ({key})", fontsize=14)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    plt.show()
    save_plot_to_path(
        fig,
        save_dir,
        f"{key}_{i}_{transform.__class__.__name__}_histogram",
        dpi=300,
        file_format=".pdf",
        transparent=True,
    )
    plt.close(fig)


def run_and_visualize_transforms(
    transforms: list[Any], sample: dict[str, Any], save_dir: Path, target_key: str
) -> None:
    """
    Apply a sequence of transforms to a sample and visualize the results.

    This function iterates through a list of transforms, applies each transform
    to the input sample, and visualizes the specified `target_key` only if the
    transform modifies it. The visualizations are saved to the specified directory.

    Args:
        transforms (List[Any]): A list of transform objects to apply to the sample.
        sample (Dict[str, Any]): A dictionary representing the input data
        save_dir (Path): The directory where visualizations will be saved.
        target_key (str): The key in the sample dictionary to visualize (e.g., 'raw_bf').
    """
    for i, transform in enumerate(transforms):
        logger.info(f"\nApplying Transform {i+1}: {transform.__class__.__name__}")

        # Determine which keys this transform operates on
        transform_keys = getattr(transform, "keys", getattr(transform, "key", []))
        if isinstance(transform_keys, str):
            transform_keys = [transform_keys]

        if target_key not in transform_keys:
            logger.info(f"  Transform does not touch '{target_key}'; skipping visualization.")
            # Still apply the transform
            sample = transform(sample)
            continue

        # Apply transform
        sample = transform(sample)

        # Visualize target_key if present
        if isinstance(sample, dict) and target_key in sample:
            value = sample[target_key]
            if isinstance(value, torch.Tensor):
                value_np = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                value_np = value
            else:
                continue

            if value_np.ndim in [2, 3]:
                save_thumbnail_to_path(
                    value_np.squeeze() if value_np.ndim == 3 else value_np,
                    f"{target_key}_{i}_{transform.__class__.__name__}",
                    save_dir,
                    figsize=(6, 6),
                    scalebar_size_um=50,
                    pixel_size=PIXEL_SIZE_3i_20x,
                )
                plot_and_save_histogram(value_np, transform, target_key, save_dir, i)

        # Handle list outputs (like RandSpatialCropSamplesd)
        elif isinstance(sample, list):
            for crop_idx, crop in enumerate(sample):
                if target_key not in crop:
                    continue
                value = crop[target_key]
                if isinstance(value, torch.Tensor):
                    value_np = value.detach().cpu().numpy()
                elif isinstance(value, np.ndarray):
                    value_np = value
                else:
                    continue
                if value_np.ndim in [2, 3]:
                    save_thumbnail_to_path(
                        value_np.squeeze() if value_np.ndim == 3 else value_np,
                        f"{target_key}_{i}_{transform.__class__.__name__}_crop{crop_idx}",
                        save_dir,
                        figsize=(6, 6),
                        scalebar_size_um=10,
                        pixel_size=PIXEL_SIZE_3i_20x,
                        bar_thickness=3,
                        bar_padding=5,
                    )
                    plot_and_save_histogram(value_np, transform, target_key, save_dir, i)

            # Stop here to avoid ToTensord errors
            break

        else:
            logger.warning(f"Unexpected sample type {type(sample)}; skipping visualization.")
            break
