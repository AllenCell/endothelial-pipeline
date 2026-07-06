import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from monai.data import MetaTensor

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_XSMALL
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1

logger = logging.getLogger(__name__)


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
        logger.debug("Applying Transform %s: %s", i + 1, {transform.__class__.__name__})

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
    output_key: str,
    figure_size: tuple = (1.5, 1.5),
    col_titles: list[str] | None = None,
    row_title: str | None = None,
) -> np.ndarray:
    """
    Apply a sequence of transforms to a sample and visualize all steps
    as a contact sheet (one row of images, one row of histograms).

    Args:
        transforms (List[Any]): Sequence of transforms to apply.
        sample (Dict[str, Any]): Input data dictionary.
        save_dir (Path): Directory where visualizations are saved.
        target_key (str): Key to visualize (e.g., 'raw_bf').
        figure_size (tuple): Size per panel in inches (width, height).
        col_titles (list[str] | None): Custom column titles. If None, uses transform class names.
        row_title (str | None): Row label (e.g., 'BF' or 'VE-cadherin').

    Returns:
        The transformed image as a NumPy array.
    """
    images: list[np.ndarray] = []
    _col_titles: list[str] = []

    for step_idx, transform in enumerate(transforms):
        transform_name = transform.__class__.__name__
        logger.info("Applying Transform %d: %s", step_idx + 1, transform_name)

        # Determine which keys this transform operates on
        transform_keys = getattr(transform, "keys", getattr(transform, "key", []))
        if isinstance(transform_keys, str):
            transform_keys = [transform_keys]

        sample = transform(sample)

        # Collect if target_key affected
        if target_key in transform_keys and isinstance(sample, dict):
            value_np = get_target_image_from_sample(sample, target_key)
            value_np = crop_image(
                value_np, start_x=0, start_y=0, crop_size=500
            )  # Crop for visualization
            images.append(value_np.squeeze())
            _col_titles.append(transform_name)

    # --- Image contact sheet ---
    n = len(images)
    # Use custom col_titles if provided, otherwise fall back to transform names
    titles = col_titles if col_titles is not None else _col_titles
    fig_images = make_contact_sheet(
        panels=images,
        max_rows=1,
        max_cols=n,
        col_titles=titles,
        row_titles=[row_title] if row_title else None,
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0, "hspace": 0},
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
    )

    for ax in fig_images.axes:
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3

    # Add scalebar to first panel only
    scale_bar_um = 100
    for i, ax in enumerate(fig_images.axes):
        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            color="white",
            label_fontsize=FONTSIZE_XSMALL,
            bar_thickness=10,
            padding=20,
            label_xy=(0.96, 0.08),
            include_label=True if i == 0 else False,
        )

    save_plot_to_path(
        fig_images,
        save_dir,
        f"{output_key}_{target_key}_images",
        file_format=".svg",
        pad_inches=0,
        tight_layout=False,
    )

    # --- Histogram contact sheet ---
    # Same total width as images, taller aspect ratio
    fig_hist, axes_hist = plt.subplots(1, n, figsize=figure_size, layout="constrained", sharey=True)
    if n == 1:
        axes_hist = [axes_hist]

    for i, img in enumerate(images):
        ax = axes_hist[i]
        ax.hist(img.ravel(), bins=50, color="grey", alpha=0.7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        if i == 0:
            ax.set_xlabel("Intensity (a.u)")
            ax.set_ylabel("Frequency")
            ax.xaxis.labelpad = 3
            ax.yaxis.labelpad = 3
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

    save_plot_to_path(
        fig_hist,
        save_dir,
        f"{output_key}_{target_key}_histograms",
        file_format=".svg",
        tight_layout=False,
    )

    transformed_image = get_target_image_from_sample(sample, target_key)
    return transformed_image
