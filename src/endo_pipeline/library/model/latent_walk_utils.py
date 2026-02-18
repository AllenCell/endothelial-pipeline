import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from endo_pipeline.library.model.diffae import DiffusionAutoEncoder, generate_from_coords

if TYPE_CHECKING:
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder

logger = logging.getLogger(__name__)


def get_latent_walk(
    dataframe: pd.DataFrame,
    n_dims: int,
    sigma: float | None,
    n_steps: int,
    replace_mean_with_pc_value: list[float | None] | None = None,
) -> tuple[np.ndarray, list]:
    """
    Generate a latent walk based on standard deviation or min/max of each
    dimension.

    Parameters
    ----------
    data
        Numpy array containing the data to be traversed.
    n_dims
        Number of dimensions for the latent walk.
    sigma
        Range of values for the latent walk. If None, use min/max of dimension.
    n_steps
        Number of steps in the latent walk.
    replace_mean_with_pc_value
        List of PC values to replace the mean with for each PC dimension. Must be of length n_dims.
        If None, uses the mean of the data.
    """
    replace_values = (
        [None] * n_dims if replace_mean_with_pc_value is None else replace_mean_with_pc_value
    )

    if len(replace_values) != n_dims:
        raise ValueError(f"Expected replace_values of length {n_dims}, got {len(replace_values)}.")

    walks: list[np.ndarray] = []
    ranges: list[np.ndarray] = []

    data = dataframe.to_numpy()
    for dim in range(n_dims):
        if sigma is None:
            data_min = data[:, dim].min()
            data_max = data[:, dim].max()
            range_ = np.linspace(data_min, data_max, n_steps)
        else:
            std = data[:, dim].std()
            range_ = np.arange(-sigma, sigma + 0.01) * std

        # Get baseline values for all dimensions as either the mean value of the
        # dimension or the given replacement value for that dimension.
        walk_values = [
            data[:, i].mean() if replace is None else replace
            for i, replace in enumerate(replace_values)
        ]

        # Stack the baseline values for all steps and then replace only the current
        # dimension with the selected latent walk values.
        dim_traversal = np.stack([walk_values] * range_.shape[0])
        dim_traversal[:, dim] = range_

        walks.append(dim_traversal)
        ranges.append(range_)

    walk_array = np.concatenate(walks).squeeze()

    return walk_array, ranges


def generate_latent_walk_images(
    model: "DiffusionAutoEncoder",
    walk: np.ndarray,
    ranges: list,
    n_noise_samples: int = 1,
    num_gpus: int | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    # Ggenerate images from the latent walk
    walk_img = generate_from_coords(
        model, walk, n_noise_samples=n_noise_samples, num_gpus=num_gpus, random_seed=random_seed
    )

    # Reshape to (n_dim, n_steps, img_w, img_h)
    n_dim = len(ranges)
    n_steps_actual = ranges[0].shape[0]
    image_width = walk_img.shape[-2]
    image_height = walk_img.shape[-1]

    return walk_img.reshape(n_dim, n_steps_actual, image_width, image_height)
