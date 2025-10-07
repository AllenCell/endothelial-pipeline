import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from endo_pipeline.io.output import save_plot_to_path, save_thumbnail_to_path
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

logger = logging.getLogger(__name__)


def plot_and_save_histogram(
    value_np: np.ndarray, transform: Any, key: str, save_dir: Path, i: int
) -> None:

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(value_np.ravel(), bins=50, color="grey", alpha=0.7)
    ax.set_title(f"{transform.__class__.__name__} ({key})", fontsize=14)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    # ax.grid(True)
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


def initialize_transform(transform_cfg):
    # Extract the module and class name from the _target_ field
    target_path = transform_cfg["_target_"]
    module_name, class_name = target_path.rsplit(".", 1)

    # Dynamically import the module and get the class
    module = importlib.import_module(module_name)
    target_class = getattr(module, class_name)

    # Extract parameters and initialize the transform
    params = {k: v for k, v in transform_cfg.items() if k != "_target_"}
    return target_class(**params)


def run_and_visualize_transforms(transforms, sample, save_dir, target_key):
    """
    Apply each transform in sequence to the sample, but only visualize
    the specified target_key if the transform actually modifies it.
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
