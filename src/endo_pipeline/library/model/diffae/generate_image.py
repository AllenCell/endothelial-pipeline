from pathlib import Path

import numpy as np
import torch

from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import ModelConfig
from src.endo_pipeline.library.model.mlflow import load_mlflow_model


def generate_from_coords(
    model_config: ModelConfig,
    coords: np.ndarray | list[list[float]],
    n_noise_samples: int = 1,
    average: bool = False,
) -> np.ndarray:
    """
    Generate a synthetic image from a list of coordinates
    in the latent space of a model.

    Parameters
    ----------
    model_name: str
        The name of the model to use for generation.
    coords: List[List[float]]
        A list of coordinates in the latent space of the model.
    n_noise_samples: int
        The number of noise samples to use for generation.
    average: bool
        Whether to average the generated images.
    """
    if not isinstance(coords, np.ndarray):
        if isinstance(coords, list):
            coords_np = np.array(coords)
        else:
            raise ValueError("coords must be a numpy array or a list of lists")
    else:
        coords_np = coords
    mlflow_id = model_config.mlflow_run_id
    model_path = Path(get_output_path(f"models/{model_config.name}"))
    model = load_mlflow_model(mlflow_id, save_path=model_path)

    coords_torch = torch.from_numpy(coords_np).float()

    # move model and inputs to gpu if available
    if torch.cuda.is_available():
        coords_ = coords_torch.to("cuda")
        model_ = model.to("cuda")
    else:
        coords_ = coords_torch
        model_ = model

    walk_img = model_.generate_from_latent(
        coords_, n_noise_samples=n_noise_samples, average=average, save=False
    )
    return walk_img


def generate_from_coords_batch(
    model_config: ModelConfig, coords_batch: np.ndarray | list[list[list[float]]]
) -> list[np.ndarray]:
    """
    Generate synthetic images from a batch of coordinates
    in the latent space of a model.

    Parameters
    ----------
    model_name: str
        The name of the model to use for generation.
    coords_batch: List[List[List[float]]]
        A batch of lists of coordinates in the latent space of the model.
    """

    coords_concat = np.concatenate(coords_batch, axis=0)
    img = generate_from_coords(model_config, coords=coords_concat)
    walk_imgs = np.split(img, len(coords_batch))

    return walk_imgs
