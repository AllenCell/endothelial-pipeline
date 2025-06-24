from pathlib import Path
from typing import List

import numpy as np
import torch

from cellsmap.util.dataset_io import get_model_info
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.model.mlflow import load_mlflow_model


def generate_from_coords(
    model_name: str,
    coords: List[List[float]],
    n_noise_samples: int = 1,
    average: bool = False,
) -> np.ndarray:
    """
    Generates a synthetic image from a list of coordinates in the latent space of a model.
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
    coords = np.array(coords)
    mlflow_id = get_model_info(model_name)["mlflow_run_id"]
    model_path = Path(get_output_path(f"models/{model_name}"))
    model = load_mlflow_model(mlflow_id, save_path=model_path)

    coords = torch.from_numpy(coords).float()

    # move model and inputs to gpu if available
    if torch.cuda.is_available():
        coords = coords.to("cuda")
        model = model.to("cuda")

    walk_img = model.generate_from_latent(
        coords, n_noise_samples=n_noise_samples, average=average, save=False
    )
    return walk_img


def generate_from_coords_batch(
    model_name: str, coords_batch: List[List[List[float]]]
) -> tuple[np.ndarray]:
    """
    Generates synthetic images from a batch of coordinates in the latent space of a model.
    Parameters
    ----------
    model_name: str
        The name of the model to use for generation.
    coords_batch: List[List[List[float]]]
        A batch of lists of coordinates in the latent space of the model.
    """

    coords_concat = np.concatenate(coords_batch, axis=0)
    img = generate_from_coords(model_name, coords=coords_concat)
    walk_imgs = np.split(img, len(coords_batch))

    return walk_imgs
