import logging
from typing import cast

import numpy as np
import pandas as pd
import torch

from src.endo_pipeline.configs import CytoDLModelConfig, load_model_config
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_manifest import get_feature_column_names
from src.endo_pipeline.library.model.mlflow_utils import load_mlflow_model

logger = logging.getLogger(__name__)


def generate_from_coords(
    model_name: str,
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

    model_config = cast(CytoDLModelConfig, load_model_config(model_name))
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_name, "train", include_timestamp=False)
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
    model_name: str, coords_batch: np.ndarray | list[list[list[float]]]
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
    logger.debug("Concatenated coordinates shape: [ %s ]", coords_concat.shape)
    img = generate_from_coords(model_name, coords=coords_concat)
    walk_imgs = np.split(img, len(coords_batch))

    return walk_imgs


def get_reconstructed_crops_in_dataframe(df: pd.DataFrame) -> list:
    """
    Reconstruct crops from each latent coordinate
    given in the input dataframe.
    """
    # get coordinates (feature columns) from the dataframe,
    # convert to list of lists for input into DiffAE model
    num_points = df.shape[0]
    latent_coords = []
    feat_cols = get_feature_column_names(df)
    for i in range(num_points):
        latent_coords.append(df[feat_cols].iloc[i].tolist())

    # pass into DiffAE model to generate reconstructed crops
    walk_img = generate_from_coords_batch(
        "diffae_04_10", np.array(latent_coords)
    )  # output is a numpy array: (# coords x 128 x 128), greyscale image

    # convert to list of numpy arrays
    walk_img_list = []
    for i in range(num_points):
        walk_img_list.append(walk_img[i])

    return walk_img_list
