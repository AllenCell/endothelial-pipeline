import logging
import typing

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate

if typing.TYPE_CHECKING:
    from cyto_dl.api import CytoDLModel

from endo_pipeline.library.analyze.diffae_dataframe import get_feature_column_names

logger = logging.getLogger(__name__)


def generate_from_coords(
    model: "CytoDLModel",
    coords: np.ndarray | list[list[float]],
    n_noise_samples: int = 1,
    average: bool = False,
) -> np.ndarray:
    """
    Generate a synthetic image from a list of coordinates
    in the latent space of a model.

    Parameters
    ----------
    model
        The model to use for generation.
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
            logger.error("Parameter `coords` must be a numpy array or a list of lists.")
            raise ValueError("coords must be a numpy array or a list of lists")
    else:
        coords_np = coords

    coords_torch = torch.from_numpy(coords_np).float()

    # have to instantiate the actual model object from the config
    model_instantiated = instantiate(model.cfg.model)

    # move model and inputs to gpu if available
    if torch.cuda.is_available():
        coords_ = coords_torch.to("cuda")
        model_ = model_instantiated.to("cuda")
    else:
        coords_ = coords_torch
        model_ = model_instantiated

    walk_img = model_.generate_from_latent(
        coords_, n_noise_samples=n_noise_samples, average=average, save=False
    )
    return walk_img


def generate_from_coords_batch(
    model: "CytoDLModel", coords_batch: np.ndarray | list[list[list[float]]]
) -> list[np.ndarray]:
    """
    Generate synthetic images from a batch of coordinates
    in the latent space of a model.

    Parameters
    ----------
    model:
        The model to use for generation.
    coords_batch:
        A batch of lists of coordinates in the latent space of the model.
    """

    # note to self: need to debug what the input type is here
    # I think the outlier is the latent walk? or maybe the crop
    # reconstruction? need to check
    if isinstance(coords_batch, np.ndarray):
        coords_concat = coords_batch.copy()
    elif isinstance(coords_batch, list):
        coords_concat = np.array(coords_batch)
    else:
        coords_concat = np.concatenate(coords_batch, axis=0)
    logger.debug("Concatenated coordinates shape: [ %s ]", coords_concat.shape)

    img = generate_from_coords(model, coords=coords_concat)
    walk_imgs = [img[i] for i in range(len(coords_batch))]

    return walk_imgs


def get_reconstructed_crops_in_dataframe(
    df: pd.DataFrame,
    model: "CytoDLModel",
) -> list:
    """Reconstruct crops from each latent coordinate given in the input dataframe."""
    # get coordinates (feature columns) from the dataframe,
    # convert to list of lists for input into DiffAE model
    num_points = df.shape[0]
    latent_coords = []
    feat_cols = get_feature_column_names(df)
    for i in range(num_points):
        latent_coords.append(df[feat_cols].iloc[i].tolist())

    # pass into DiffAE model to generate reconstructed crops
    walk_imgs = generate_from_coords_batch(
        model, np.array(latent_coords)
    )  # output is a list of numpy arrays

    return walk_imgs
