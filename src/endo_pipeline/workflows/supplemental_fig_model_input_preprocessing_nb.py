# %%
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
from endo_pipeline.io.output import get_output_path, save_plot_to_path, save_thumbnail_to_path
from endo_pipeline.library.model import get_model_dir
from endo_pipeline.settings.examples import EXAMPLE_DATASET

# %% YAML Config Path
# Dynamically load the YAML file from the package
yaml_path = get_model_dir() / "diffae_training.yaml"
config = OmegaConf.load(yaml_path)

# Access the training transform configuration
train_transform_cfg = config.data.train_dataloaders.dataset.transform
temp_squeeze = {"_target_": "monai.transforms.SqueezeDimd", "keys": "raw", "dim": 0}
# Insert into the list of transforms (not the Compose object itself)
train_transform_cfg.transforms.insert(1, OmegaConf.create(temp_squeeze))


# %% Image Path
dataset = EXAMPLE_DATASET["SUP_FIG_Y"]

print(f"Using dataset {dataset}")
position = 0
config = load_dataset_config(dataset)
test_image_path = get_zarr_file_for_position(config, position)

save_dir = get_output_path("model_input_preprocessing_viz", f"{dataset}_P{position}")

# %%
sample = {"original_path": str(test_image_path)}


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


# Initialize all transforms in the pipeline
transforms = [initialize_transform(t) for t in train_transform_cfg.transforms]


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
        f"{i}_{transform.__class__.__name__}_{key}_histogram",
        dpi=300,
        file_format=".pdf",
        transparent=True,
    )
    plt.close(fig)


# Step through each transform
for i, transform in enumerate(transforms):
    print(f"\nApplying Transform {i+1}: {transform.__class__.__name__}")

    # Apply transform
    sample = transform(sample)

    # --- Handle list outputs (like RandSpatialCropSamplesd) ---
    if isinstance(sample, list):
        print(f"Transform {transform.__class__.__name__} produced {len(sample)} crops")
        for idx, crop in enumerate(sample):
            print(f"Cropped sample {idx}: keys = {list(crop.keys())}")
            for key, value in crop.items():
                if isinstance(value, torch.Tensor):
                    value_np = value.detach().cpu().numpy()
                elif isinstance(value, np.ndarray):
                    value_np = value
                else:
                    continue
                if value_np.ndim in [2, 3]:
                    save_thumbnail_to_path(
                        value_np.squeeze() if value_np.ndim == 3 else value_np,
                        f"{i}_{transform.__class__.__name__}_{key}_crop{idx}",
                        save_dir,
                        figsize=(6, 6),
                    )
                    plot_and_save_histogram(value_np, transform, key, save_dir, i)

        # Stop here to avoid ToTensord error
        break

    # --- Handle dict outputs ---
    elif isinstance(sample, dict):
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                value_np = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                value_np = value
            else:
                continue
            if value_np.ndim in [2, 3]:
                save_thumbnail_to_path(
                    value_np.squeeze() if value_np.ndim == 3 else value_np,
                    f"{i}_{transform.__class__.__name__}_{key}",
                    save_dir,
                    figsize=(6, 6),
                )
                plot_and_save_histogram(value_np, transform, key, save_dir, i)

    else:
        print(f"Unexpected sample type {type(sample)}, skipping visualization")
        break


# %%
