import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_latent_feature_column_names,
    get_pc_column_names,
)
from endo_pipeline.library.model.diffae import generate_from_coords
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

if TYPE_CHECKING:
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder

logger = logging.getLogger(__name__)


def get_max_dim_in_column_names(column_names: list[str], feature_prefix: str) -> int:
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
    if len(dims) == 0:
        raise ValueError(f"No column names found with prefix '{feature_prefix}' in {column_names}.")

    return max(dims)


def get_num_pcs_from_column_names(column_names: list[str]) -> int:
    """
    Get the number of principal components needed for the latent walk based on
    the provided column names.

    Parameters
    ----------
    column_names
        List of column names corresponding to each dimension.
    """
    # get the maximum PC dimension number from the column names; if no PC
    # dimensions are included, set max_pc_dim to 0
    try:
        max_pc_dim = get_max_dim_in_column_names(column_names, ColumnName.PCA_FEATURE_PREFIX.value)
    except ValueError:
        max_pc_dim = 0

    # check special case for polar coordinates, which at minimum need the first
    # two PC dimensions to be included
    if (
        ColumnName.POLAR_ANGLE.value in column_names
        or ColumnName.POLAR_RADIUS.value in column_names
    ):
        max_pc_dim = max(max_pc_dim, 2)

    # check if PC3_FLIPPED is included, which also requires at minimum the first
    # three PC dimensions to be included
    if ColumnName.PC3_FLIPPED.value in column_names:
        max_pc_dim = max(max_pc_dim, 3)

    return max_pc_dim


def _add_preceding_dims_to_column_names(column_names: list[str], feature_prefix: str) -> list[str]:
    """
    Add preceding dimension column names to the provided column names based on
    the provided feature prefix.

    For example, column_names = ["pc_3"] and feature_prefix = "pc_", this method
    would add "pc_1" and "pc_2" to the output list of column names since they
    are the preceding dimensions for "pc_3".
    """
    try:
        # get max dimension number for the given feature prefix from the column names
        max_dim = get_max_dim_in_column_names(column_names, feature_prefix)

        # get all column names for the preceding dimensions based on the feature
        # prefix and max dimension number
        if feature_prefix == ColumnName.PCA_FEATURE_PREFIX:
            all_dim_columns = get_pc_column_names(num_pcs=max_dim)
        elif feature_prefix == ColumnName.LATENT_FEATURE_PREFIX:
            all_dim_columns = get_latent_feature_column_names(num_latent_dims=max_dim)
        else:
            raise ValueError(f"Invalid feature prefix: {feature_prefix}")
        # combine the original column names with the preceding dimension column
        # names, ensuring no duplicates
        column_names_with_preceding_dims = list(set(column_names + all_dim_columns))
        return column_names_with_preceding_dims
    except ValueError:
        logger.warning(
            "No column names found with prefix [ %s ] in [ %s ]. No preceding dimensions will be added.",
            feature_prefix,
            column_names,
        )
        return column_names


def get_column_names_for_latent_walk_dataframe(input_column_names: list[str]) -> list[str]:
    """
    Set up column names for the latent walk based on the provided column names.

    For example, if the provided columns include any of the PC features, the
    preceding PC features must also be present in the dataframe so that their
    values can be used as the baseline for the latent walk. So if the provided
    column names include PC3, then the column names for the latent walk should
    include PC1, PC2, and PC3.

    In the case that PC3 is included but neither {PC1, PC2} nor {polar angle,
    polar radius} are included, then the column names for the PC1 and PC2 are
    added (as opposed to the polar angle and radius). Either are sufficient for
    calculating the PC1, PC2, and PC3 coordinates for image generation, but the
    PC1 and PC2 coordinates are used as the baseline for the latent walk in this
    case since they are more commonly used as the baseline for latent walks in
    general.

    Similarly, if the provided column names include any of the latent features,
    the preceding latent features must also be present in the dataframe so that
    their values can be used as the baseline for the latent walk.

    If either of the polar coordinate columns are included, both must be
    included, so that the inverse transform can be applied to convert back to
    Cartesian coordinates for image generation.
    """
    column_names = input_column_names.copy()
    # special cases for transformed variables: polar coordinates and flipped pc3
    polar_subset = {ColumnName.POLAR_ANGLE.value, ColumnName.POLAR_RADIUS.value}
    pc1_pc2_subset = {
        f"{ColumnName.PCA_FEATURE_PREFIX}1",
        f"{ColumnName.PCA_FEATURE_PREFIX}2",
    }
    # first, check that columns do not have both the polar coordinates and the
    # PC1 and PC2 coordinates, since this would be redundant and could cause
    # issues with image generation since the polar coordinates would not be able
    # to be used for the inverse PCA transformation to get the Cartesian
    # coordinates for image generation
    if polar_subset.issubset(column_names) and pc1_pc2_subset.issubset(column_names):
        raise ValueError(
            f"Column names cannot include both polar coordinates and PC1 and PC2 coordinates. Column names provided: {column_names}"
        )
    if ColumnName.POLAR_ANGLE in column_names and ColumnName.POLAR_RADIUS not in column_names:
        # if polar angle is included in the column names but polar radius is
        # not, add polar radius to the column names
        column_names.append(ColumnName.POLAR_RADIUS.value)
    if ColumnName.POLAR_RADIUS in column_names and ColumnName.POLAR_ANGLE not in column_names:
        # if polar radius is included in the column names but polar angle is
        # not, add polar angle to the column names
        column_names.append(ColumnName.POLAR_ANGLE.value)
    if ColumnName.PC3_FLIPPED in column_names:
        # if PC3_FLIPPED is included in the column names, need to either
        # have PC1 and PC2 OR polar angle and radius included in the column names
        # so that the PC1, PC2, and PC3 coordinates can be calculated for image generation
        if not polar_subset.issubset(column_names) and not pc1_pc2_subset.issubset(column_names):
            # if neither the polar coordinate columns nor the PC1 and PC2 columns are included in the column names, add the PC1 and PC2 columns to the column names
            column_names = _add_preceding_dims_to_column_names(
                column_names, ColumnName.PCA_FEATURE_PREFIX.value
            )

    # add preceding latent feature columns if any latent feature columns are included in the column names
    column_names = _add_preceding_dims_to_column_names(
        column_names, ColumnName.LATENT_FEATURE_PREFIX.value
    )
    column_names = _add_preceding_dims_to_column_names(
        column_names, ColumnName.PCA_FEATURE_PREFIX.value
    )

    return column_names


def get_baseline_walk_values(
    dataframe: pd.DataFrame,
    set_column_value: dict[str, float] | None = None,
) -> list[float]:
    """
    Get baseline walk values for each dimension based on the mean of the data or
    provided replacement values.

    Parameters
    ----------
    dataframe
        DataFrame containing the data to calculate mean values from.
    set_column_value
        Optional, dictionary mapping column names to values to set for those
        columns when generating the latent walk. If None, uses the mean of the
        data for each dimension as the baseline walk values.

    Returns
    -------
    :
        List of baseline walk values for each dimension.
    """

    baseline_values = []
    for column_name in dataframe.columns:
        if (set_column_value is None) or (column_name not in set_column_value.keys()):
            baseline_values.append(dataframe[column_name].mean())
        else:
            baseline_values.append(set_column_value[column_name])

    return baseline_values


def get_latent_walk(
    dataframe: pd.DataFrame,
    walk_column_names: list[str],
    sigma: float | None,
    n_steps: int,
    set_column_value: dict[str, float] | None = None,
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
    set_column_value
        Optional, dictionary mapping column names to values to set for those
        columns when generating the latent walk. If None, uses the mean of the
        data for each dimension as the baseline walk values.
    """
    walks: list[pd.DataFrame] = []
    ranges: list[np.ndarray] = []

    # Get baseline values for all dimensions as either the mean value of the
    # dimension or the given replacement value for that dimension.
    baseline_walk_values = get_baseline_walk_values(dataframe, set_column_value)

    for column_name in walk_column_names:
        walk_data = dataframe[column_name]
        if sigma is None:
            data_min = walk_data.min()
            data_max = walk_data.max()
            range_ = np.linspace(data_min, data_max, n_steps)
        else:
            if sigma <= 0:
                raise ValueError(f"Input sigma must be positive, got {sigma}.")
            std = walk_data.std()
            range_ = np.arange(-sigma, sigma + 0.01) * std

        # Stack the baseline values for all steps and then replace only the current
        # dimension with the selected latent walk values.
        dim_traversal_array = np.stack([baseline_walk_values] * range_.shape[0])
        dim_traversal_df = pd.DataFrame(dim_traversal_array, columns=dataframe.columns)
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
