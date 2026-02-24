import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from endo_pipeline.library.model.diffae import generate_from_coords
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

if TYPE_CHECKING:
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder

logger = logging.getLogger(__name__)


def _get_max_dim_in_column_names(column_names: list[str], feature_prefix: str) -> int:
    """
    Get the maximum number of dimensions from the provided column names.

    Parameters
    ----------
    column_names
        List of column names corresponding to each dimension.
    feature_prefix
        Prefix to look for in column names, e.g., "pc_" or "feat_".

    Returns
    -------
    int
        Maximum number of dimensions based on the column names.
    """
    # Define pattern that starts with feature prefix followed by 1 or more digits
    pattern = rf"{feature_prefix}(\d+)"

    # Apply match to each column name in list
    matches = [re.match(pattern, column) for column in column_names]

    # Iterate through valid matches and convert capture group to integer
    dims = [int(match.group(1)) for match in matches if match]

    return max(dims)


def get_num_dims_from_column_names(column_names: list[str]) -> int:
    """
    Get the number of dimensions for the latent walk based on the provided
    column names.

    Parameters
    ----------
    column_names
        List of column names corresponding to each dimension.

    Returns
    -------
    int
        Number of dimensions for the latent walk.
    """
    # depending on whether column names are for PCA features or latent features,
    # get the maximum dimension number from the column names
    max_pc_dim = _get_max_dim_in_column_names(column_names, ColumnName.PCA_FEATURE_PREFIX.value)
    max_latent_dim = _get_max_dim_in_column_names(
        column_names, ColumnName.LATENT_FEATURE_PREFIX.value
    )

    max_dim = max(max_pc_dim, max_latent_dim)

    # check special case for polar coordinates, which at minimum need the first
    # two PC dimensions to be included
    if (
        ColumnName.POLAR_ANGLE.value in column_names
        or ColumnName.POLAR_RADIUS.value in column_names
    ):
        max_dim = max(max_dim, 2)

    # check if PC3_FLIPPED is included, which also requires at minimum the first
    # three PC dimensions to be included
    if ColumnName.PC3_FLIPPED.value in column_names:
        max_dim = max(max_dim, 3)

    return max_dim


def get_baseline_walk_values(
    dataframe: pd.DataFrame,
    column_names: list[str],
    replace_mean_with_val: list[float] | None = None,
) -> list[float]:
    """
    Get baseline walk values for each dimension based on the mean of the data or
    provided replacement values.

    Parameters
    ----------
    dataframe
        DataFrame containing the data to calculate mean values from.
    column_names
        List of column names corresponding to each dimension.
    replace_mean_with_val
        Optional, list of values to replace the mean with for each dimension. If
        None, uses the mean of the data for each dimension.

    Returns
    -------
    :
        List of baseline walk values for each dimension.
    """
    n_dims = len(column_names)

    # convert replace_mean_with_val to a list of length n_dims, filling with None if it is None
    replace_values = [None] * n_dims if replace_mean_with_val is None else replace_mean_with_val

    if len(replace_values) != n_dims:
        raise ValueError(
            f"Expected replace_mean_with_val of length {len(column_names)}, got {len(replace_values)}."
        )

    baseline_values = []
    for col_name, replace_value in zip(column_names, replace_values, strict=True):
        if replace_value is None:
            baseline_values.append(dataframe[col_name].mean())
        else:
            baseline_values.append(replace_value)

    return baseline_values


def get_latent_walk(
    dataframe: pd.DataFrame,
    column_names: list[str],
    sigma: float | None,
    n_steps: int,
    replace_mean_with_val: list[float] | None = None,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
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
    replace_mean_with_val
        Optional, list of values to replace the mean with for each dimension
        when generating the latent walk. If None, uses the mean of the data.
    """
    walks: list[pd.DataFrame] = []
    ranges: list[np.ndarray] = []

    # Get baseline values for all dimensions as either the mean value of the
    # dimension or the given replacement value for that dimension.
    baseline_walk_values = get_baseline_walk_values(dataframe, column_names, replace_mean_with_val)

    for column_name in column_names:
        data = dataframe[column_name]
        if sigma is None:
            data_min = data.min()
            data_max = data.max()
            range_ = np.linspace(data_min, data_max, n_steps)
        else:
            if sigma <= 0:
                raise ValueError(f"Input sigma must be positive, got {sigma}.")
            std = data.std()
            range_ = np.arange(-sigma, sigma + 0.01) * std

        # Stack the baseline values for all steps and then replace only the current
        # dimension with the selected latent walk values.
        dim_traversal_array = np.stack([baseline_walk_values] * range_.shape[0])
        dim_traversal_df = pd.DataFrame(dim_traversal_array, columns=column_names)
        dim_traversal_df[column_name] = range_

        walks.append(dim_traversal_df)
        ranges.append(range_)

    walk_dataframe = pd.concat(walks, ignore_index=True)

    return walk_dataframe, ranges


def generate_latent_walk_images(
    model: "DiffusionAutoEncoder",
    walk: np.ndarray,
    ranges: list[np.ndarray],
    n_noise_samples: int = 1,
    num_gpus: int | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Generate images from a latent walk using the provided model.

    Parameters
    ----------
    model
        Model to use for image generation.
    walk
        Numpy array of shape (n_steps, n_dim) containing the latent walk
        coordinates.
    ranges
        List of numpy arrays containing the ranges of values for each dimension
        in the walk.
    n_noise_samples
        Number of noise samples to use for generating images.
    num_gpus
        Number of GPUs to use for image generation. If None, uses CPU.
    random_seed
        Random seed for reproducibility of image generation. If None, does not
        set a random seed.

    Returns
    -------
    :
        Array of stacked generated images from the latent walk, reshaped to
        (n_dim, n_steps, img_width, img_height).
    """
    walk_img = generate_from_coords(
        model, walk, n_noise_samples=n_noise_samples, num_gpus=num_gpus, random_seed=random_seed
    )

    # Reshape to (n_dim, n_steps, img_w, img_h)
    n_dim = len(ranges)
    n_steps_actual = ranges[0].shape[0]
    image_width = walk_img.shape[-2]
    image_height = walk_img.shape[-1]

    return walk_img.reshape(n_dim, n_steps_actual, image_width, image_height)
