"""Methods for generating a latent walk and corresponding images from a DiffusionAutoEncoder model."""

import logging
import re

import numpy as np
import pandas as pd

from endo_pipeline.library.analyze.polar_coords import polar_to_pcs
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
)

logger = logging.getLogger(__name__)


def get_feature_coordinates_as_string(
    column_names: list[str], feature_coordinates: np.ndarray | list[float]
) -> str:
    """
    Get a string representation of the feature coordinates for use in file names.

    Parameters
    ----------
    column_names
        List of column names corresponding to each dimension.
    feature_coordinates
        Array of feature coordinates corresponding to the column names.

    Returns
    -------
    :
        String representation of the feature coordinates for the given column
        names and feature coordinates.

    """

    coordinate_strings = []
    for column_name, coordinate in zip(column_names, feature_coordinates, strict=True):
        coordinate_str = f"{coordinate:.2f}".replace(".", "p").replace("-", "neg")
        coordinate_strings.append(f"{column_name}_{coordinate_str}")
    return "_".join(coordinate_strings)


def add_pc_coordinates_to_dataframe(
    dataframe: pd.DataFrame, feature_column_names: list[str]
) -> pd.DataFrame:
    """
    Add PC coordinates to the dataframe based on the provided feature column
    names.

    If the provided feature column names include any of the polar coordinate
    columns or the flipped PC3 column, the corresponding PC coordinates will be
    calculated and added to the dataframe as new columns.

    The PC coordinates are needed for image generation since the inverse PCA
    transformation is applied to get the latent coordinates for image
    generation, and the polar coordinates and flipped PC3 coordinate cannot be
    used directly for the inverse PCA transformation.

    Parameters
    ----------
    dataframe
        DataFrame containing the data to calculate PC coordinates from.
    feature_column_names
        List of column names corresponding to input features.

    Returns
    -------
    :
        DataFrame with added PC coordinate columns if any of the polar coordinate
        or flipped PC3 columns are present.
    """
    # if polar angle and radius are included in the column names, convert them
    # to PC1 and PC2 coordinates for image generation (inverse PCA
    # transformation cannot be performed with polar coordinates)
    if (
        Column.DiffAEData.POLAR_ANGLE in feature_column_names
        and Column.DiffAEData.POLAR_RADIUS in feature_column_names
    ):
        pc1_column_name = ColumnTemplate.PCA_FEATURE % 1
        pc2_column_name = ColumnTemplate.PCA_FEATURE % 2
        angle = dataframe[Column.DiffAEData.POLAR_ANGLE].to_numpy()
        radius = dataframe[Column.DiffAEData.POLAR_RADIUS].to_numpy()
        pc1_values, pc2_values = polar_to_pcs(angle, radius)
        dataframe[pc1_column_name] = pc1_values
        dataframe[pc2_column_name] = pc2_values

    # if flipped pc3 is included in the column names, convert it to regular pc3
    # before performing inverse PCA transformation for image generation
    if Column.DiffAEData.PC3_FLIPPED in feature_column_names:
        pc3_column_name = ColumnTemplate.PCA_FEATURE % 3
        dataframe[pc3_column_name] = -dataframe[Column.DiffAEData.PC3_FLIPPED].to_numpy()

    return dataframe


def get_max_dim_in_column_names(column_names: list[str], feature_template: str) -> int:
    """Get the maximum number of dimensions from the provided column names.

    Parameters
    ----------
    column_names
        List of column names corresponding to each dimension.
    feature_template
        Template to look for in column names, e.g., "pc_%d" or "feat_%d".

    Returns
    -------
    :
        Maximum number of dimensions based on the column names.

    """
    # Define pattern that starts with feature prefix followed by 1 or more digits
    pattern = feature_template.replace("%d", r"(\d+)")

    # Apply match to each column name in list
    matches = [re.match(pattern, column) for column in column_names]

    # Iterate through valid matches and convert capture group to integer
    dims = [int(match.group(1)) for match in matches if match]
    if len(dims) == 0:
        raise ValueError(
            f"No column names found for template '{feature_template}' in {column_names}."
        )

    return max(dims)


def get_num_pcs_from_column_names(column_names: list[str]) -> int:
    """Get the number of principal components needed to walk along the given column names.

    Parameters
    ----------
    column_names
        List of column names corresponding to each dimension.

    Returns
    -------
    :
        The full set of principal component dimensions needed based on the
        provided column names.

    """
    # get the maximum PC dimension number from the column names; if no PC
    # dimensions are included, set max_pc_dim to 0
    try:
        max_pc_dim = get_max_dim_in_column_names(column_names, ColumnTemplate.PCA_FEATURE)
    except ValueError:
        max_pc_dim = 0

    # check special case for polar coordinates, which at minimum need the first
    # two PC dimensions to be included
    if (
        Column.DiffAEData.POLAR_ANGLE in column_names
        or Column.DiffAEData.POLAR_RADIUS in column_names
    ):
        max_pc_dim = max(max_pc_dim, 2)

    # check if PC3_FLIPPED is included, which also requires at minimum the first
    # three PC dimensions to be included
    if Column.DiffAEData.PC3_FLIPPED in column_names:
        max_pc_dim = max(max_pc_dim, 3)

    return max_pc_dim


def _add_preceding_dims_to_column_names(
    column_names: list[str], feature_template: str
) -> list[str]:
    """
    Add preceding dimension column names to the provided column names.

    The input feature template is used to identify the relevant column names and
    determine the maximum dimension number included in the column names for that
    feature template. Then, all preceding dimension column names based on the
    feature template and maximum dimension number are added to the provided
    column names.

    For example, column_names = ["pc_3"] and feature_template = "pc_%d", this
    method would add "pc_1" and "pc_2" to the output list of column names since
    they are the preceding dimensions for "pc_3".

    Parameters
    ----------
    column_names
        List of column names corresponding to each dimension.
    feature_template
        Template to look for in column names, e.g., "pc_%d" or "feat_%d".

    Returns
    -------
    :
        List of column names including the original column names and the
        added preceding dimension column names.

    """
    try:
        # get max dimension number for the given feature prefix from the column names
        max_dim = get_max_dim_in_column_names(column_names, feature_template)

        # get all column names for the preceding dimensions based on the feature
        # prefix and max dimension number
        if feature_template == ColumnTemplate.PCA_FEATURE:
            all_dim_columns = DIFFAE_PC_COLUMN_NAMES[:max_dim]
        elif feature_template == ColumnTemplate.LATENT_FEATURE:
            all_dim_columns = DIFFAE_FEATURE_COLUMN_NAMES[:max_dim]
        else:
            raise ValueError(f"Invalid feature template: {feature_template}")
        # combine the original column names with the preceding dimension column
        # names, ensuring no duplicates
        column_names_with_preceding_dims = list(set(column_names + all_dim_columns))
        return column_names_with_preceding_dims
    except ValueError:
        logger.warning(
            "No column names found with prefix [ %s ] in [ %s ]. No preceding dimensions will be added.",
            feature_template,
            column_names,
        )
        return column_names


def get_column_names_for_latent_walk_dataframe(input_column_names: list[str]) -> list[str]:
    """Set up column names for the latent walk based on the provided column names.

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

    Parameters
    ----------
    input_column_names
        Initial list of column names corresponding to each dimension for the
        latent walk.

    Returns
    -------
    :
        List of column names corresponding to each dimension for the latent walk,
        including any additional column names as needed.

    """
    column_names = input_column_names.copy()
    # special cases for transformed variables: polar coordinates and flipped pc3
    polar_subset = {Column.DiffAEData.POLAR_ANGLE, Column.DiffAEData.POLAR_RADIUS.value}
    pc1_pc2_subset = {
        ColumnTemplate.PCA_FEATURE % 1,
        ColumnTemplate.PCA_FEATURE % 2,
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
    if (
        Column.DiffAEData.POLAR_ANGLE in column_names
        and Column.DiffAEData.POLAR_RADIUS not in column_names
    ):
        # if polar angle is included in the column names but polar radius is
        # not, add polar radius to the column names
        column_names.append(Column.DiffAEData.POLAR_RADIUS)
    if (
        Column.DiffAEData.POLAR_RADIUS in column_names
        and Column.DiffAEData.POLAR_ANGLE not in column_names
    ):
        # if polar radius is included in the column names but polar angle is
        # not, add polar angle to the column names
        column_names.append(Column.DiffAEData.POLAR_ANGLE)
    if Column.DiffAEData.PC3_FLIPPED in column_names:
        # if PC3_FLIPPED is included in the column names, need to either
        # have PC1 and PC2 OR polar angle and radius included in the column names
        # so that the PC1, PC2, and PC3 coordinates can be calculated for image generation
        if not polar_subset.issubset(column_names) and not pc1_pc2_subset.issubset(column_names):
            # if neither the polar coordinate columns nor the PC1 and PC2 columns are included in the column names, add the PC1 and PC2 columns to the column names
            column_names = _add_preceding_dims_to_column_names(
                column_names, ColumnTemplate.PCA_FEATURE
            )

    # add preceding latent feature columns if any latent feature columns are included in the column names
    column_names = _add_preceding_dims_to_column_names(column_names, ColumnTemplate.LATENT_FEATURE)
    column_names = _add_preceding_dims_to_column_names(column_names, ColumnTemplate.PCA_FEATURE)

    return column_names


def get_baseline_walk_values(
    dataframe: pd.DataFrame,
    set_column_value: dict[str, float] | None = None,
) -> list[float]:
    """Get baseline latent walk values for each dimension.

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
            set_value = set_column_value[column_name]
            if set_value < dataframe[column_name].min() or set_value > dataframe[column_name].max():
                logger.warning(
                    "Provided baseline value [ %.3f ] for [ %s ] is out of the range of the data: [ (%.3f, %.3f) ]. Using provided value anyway.",
                    set_value,
                    column_name,
                    dataframe[column_name].min(),
                    dataframe[column_name].max(),
                )
            baseline_values.append(set_column_value[column_name])

    return baseline_values


def get_latent_walk(
    dataframe: pd.DataFrame,
    walk_column_names: list[str],
    sigma: float | None,
    n_steps: int,
    set_column_value: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a latent walk based on standard deviation or min/max of each dimension.

    **Specifying walk dimensions**

    The `walk_column_names` parameter specifies the dimensions to traverse in
    the latent walk. For example, if the column names for the PCA features are
    `pc_1`, `pc_2`, and `pc_3`, and the `walk_column_names` parameter is set to
    [`pc_3`], then the latent walk will only traverse the `pc_3` dimension while
    holding the `pc_1` and `pc_2` dimensions constant at the baseline walk
    values.

    **Baseline walk values**

    For the dimensions that are not being traversed in the walk, the values for
    those dimensions will be held constant at the baseline walk values. By
    default, the baseline walk values are set to the mean of the data for each
    dimension, but the user can also specify custom baseline walk values for any
    dimensions using the `set_column_value` parameter. This method calls the
    method `get_baseline_walk_values` to get the baseline walk values for each
    dimension based on the provided data and the `set_column_value` parameter.

    **Walk range**

    The range of values to traverse for each dimension in the walk can be set
    based on either the standard deviation of the data for that dimension or the
    minimum and maximum values of the data for that dimension.

    If the `sigma` parameter is set to a positive value, the walk range for each
    dimension will be set to [`-sigma * std`, `sigma * std`], where std is the
    standard deviation of the data for that dimension.

    If the `sigma` parameter is set to None, the walk range for each dimension
    will be set to [`min`, `max`], where `min` and `max` are the minimum and
    maximum values of the data for that dimension.

    Parameters
    ----------
    dataframe
        Data to use for calculating baseline mean values and walk ranges.
    walk_column_names
        List of column names to actually traverse in the latent walk.
    sigma
        Optional, number of standard deviations to use for the walk range.
    n_steps
        Number of steps in the latent walk.
    set_column_value
        Optional, dictionary mapping column names to set baseline values.

    Returns
    -------
    :
        Tuple containing the latent walk dataframe and the array of
        walk ranges for each dimension.

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

    ranges_array = np.vstack(ranges)

    return walk_dataframe, ranges_array
