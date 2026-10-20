"""
Methods for computing and extrapolating noise amplitude tensors from
Kramers-Moyal diffusion estimates.
"""

import logging
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    get_kernel_residual_covariance,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.vector_field_function import compute_extrapolated_vector_field
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate


def compute_noise_amplitude(sigma_squared_tensor: np.ndarray) -> np.ndarray:
    """
    Solve `2 D = sigma sigma^T` for the noise amplitude sigma at each grid point.

    The noise amplitude is taken as the lower-triangular Cholesky factor of
    `2D`. Bins whose diffusion tensor is masked (`NaN`) or not positive
    definite are left as `NaN`.

    Parameters
    ----------
    diffusion_tensor
        Symmetric diffusion tensor field with shape `(..., ndim, ndim)`.

    Returns
    -------
    :
        Noise amplitude field `sigma` with the same shape as the input.

    """
    sigma = np.full_like(sigma_squared_tensor, np.nan)
    n_invalid = 0

    for index in np.ndindex(sigma_squared_tensor.shape[:-2]):
        tensor = sigma_squared_tensor[index]
        if not np.all(np.isfinite(tensor)):
            continue
        try:
            sigma[index] = np.linalg.cholesky(tensor)
        except np.linalg.LinAlgError:
            n_invalid += 1

    if n_invalid:
        logger.debug(
            "Diffusion tensor was not positive definite at %d grid point(s); "
            "noise amplitude left as NaN there.",
            n_invalid,
        )

    return sigma


def extrapolate_noise_amplitude_tensor(sigma: np.ndarray, centers: list[np.ndarray]) -> np.ndarray:
    """
    Fill NaN entries of the noise amplitude tensor field by interpolation.

    Kernel-smoothed noise amplitude estimates are NaN wherever the local data
    density was too low (see `compute_kernel_noise_amplitude`). Differentiating a
    field with scattered NaN bins directly (see
    `compute_noise_induced_drift_residual`) propagates NaN to every
    neighboring bin via `numpy.gradient`'s central differences, which can turn
    a sparsely-masked field into an almost entirely NaN one. Filling the NaN
    bins first, the same way the drift field itself is extrapolated, avoids
    this cascade.

    Parameters
    ----------
    sigma
        Noise amplitude tensor field with shape `(..., ndim, ndim)`.
    centers
        Bin centre arrays for each dimension.

    Returns
    -------
    :
        Noise amplitude tensor field with the same shape, with NaN entries
        filled in by interpolation/extrapolation from the valid entries.
    """
    ndim = len(centers)
    grid_shape = sigma.shape[:-2]
    sigma_flat = sigma.reshape(*grid_shape, ndim * ndim)
    filled = compute_extrapolated_vector_field(
        sigma_flat, centers, method="linear", for_vtk_files=False
    )
    return np.stack(filled["vectors"], axis=-1).reshape(*grid_shape, ndim, ndim)


def compute_kernel_residual_covariance(
    positions: np.ndarray,
    residuals: np.ndarray,
    bins: list[np.ndarray],
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a kernel-smoothed estimate of the residual autocovariance,
    `<eta(t) eta(t)^T | x(t) = x>`, conditioned on the state `x(t)`.

    This uses the same kernel-convolution-based estimator as the drift vector
    field (see `compute_drift_vector_field`), applied to the already
    drift-removed residuals `eta(t)` (see `compute_centered_residuals`) instead
    of the raw forward differences, so the per-bin estimate is smoothed over
    neighboring bins rather than averaged from the (often sparse) samples
    within a single bin.

    Parameters
    ----------
    positions
        Array of shape `(num_patches, num_timepoints, ndim)` with the state
        `x(t)` for every patch/timepoint pair, NaN where unobserved.
    residuals
        Array of shape `(num_patches, num_timepoints, ndim)` with the
        drift-removed residuals `eta(t)`, NaN where unobserved.
    bins
        List of monotonically increasing bin edges for each dimension of the
        state space to condition the average on.
    kernel
        Kramers-Moyal kernel or list of kernels (one per dimension) used for
        the kernel convolution, matching the kernel(s) used for the
        pre-computed drift vector field.

    Returns
    -------
    :
        Tuple of the symmetric kernel-smoothed covariance tensor, shape
        `(*grid_shape, ndim, ndim)`, and the raw (unsmoothed) number of
        samples contributing to each bin, shape `(*grid_shape,)`, for
        diagnostic purposes.
    """
    is_valid = ~np.isnan(residuals).any(axis=-1) & ~np.isnan(positions).any(axis=-1)
    x_valid = positions[is_valid]
    eta_valid = residuals[is_valid]

    covariance = get_kernel_residual_covariance(x_valid, eta_valid, bins=bins, kernel=kernel)
    counts = np.histogramdd(x_valid, bins=bins)[0]

    return covariance, counts


def create_noise_amplitude_vector_field_df(
    noise_amplitude: np.ndarray,
    sample_counts: np.ndarray,
    column_names: list[str | Column.DiffAEData],
    feature_grid: tuple[np.ndarray, ...],
    metadata_dict: dict[str, str | float] | None = None,
) -> pd.DataFrame:
    """
    Create dataframe containing the kernel-smoothed noise amplitude tensor
    field for a single flow condition.

    The output dataframe has columns for the grid points in each dimension
    (matching the layout of `create_drift_vector_field_df`), the unique
    entries of the noise amplitude tensor `sigma_ij(x)`, and the number of
    residual samples contributing to each grid point.

    `noise_amplitude` is assumed to be the lower-triangular Cholesky factor
    produced by `compute_noise_amplitude`, so for `i < j` the entry `sigma[i,
    j]` is trivially zero while `sigma[j, i]` carries the real value; the
    column labeled with `(i, j)` is therefore populated from `sigma[j, i]`,
    not `sigma[i, j]`.

    Parameters
    ----------
    noise_amplitude
        Noise amplitude tensor field, shape `(*grid_shape, ndim, ndim)`.
    sample_counts
        Number of residual samples contributing to each bin, shape
        `(*grid_shape,)`.
    column_names
        List of column names corresponding to the features used for computing
        the noise amplitude.
    feature_grid
        Tuple of ND arrays with the grid points in each dimension, matching
        `sample_counts.shape`.
    metadata_dict
        Optional, dictionary containing metadata to include in the output
        dataframe (e.g. dataset name, shear stress).
    """
    mesh_column_names = [ColumnTemplate.MESH_GRID % name for name in column_names]
    noise_amplitude_dataframe = pd.DataFrame(columns=[Column.DATASET, *mesh_column_names])

    for mesh_column_name, grid in zip(mesh_column_names, feature_grid, strict=True):
        noise_amplitude_dataframe[mesh_column_name] = grid.flatten()

    for feature_index_1, feature_index_2 in combinations_with_replacement(
        range(len(column_names)), 2
    ):
        tensor_column_name = ColumnTemplate.NOISE_AMPLITUDE % (
            column_names[feature_index_1],
            column_names[feature_index_2],
        )
        noise_amplitude_dataframe[tensor_column_name] = noise_amplitude[
            ..., feature_index_2, feature_index_1
        ].flatten()

    noise_amplitude_dataframe[Column.NoiseAmplitude.SAMPLE_COUNT] = sample_counts.flatten()

    if metadata_dict is not None:
        for key in metadata_dict:
            noise_amplitude_dataframe[key] = metadata_dict[key]

    return noise_amplitude_dataframe


def get_reshaped_noise_amplitude_and_grid(
    noise_amplitude_dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Reconstruct the noise amplitude tensor field `sigma(x)` and its grid from
    the wide dataframe produced by `create_noise_amplitude_vector_field_df`.

    This assumes `sigma` is the lower-triangular Cholesky factor (see
    `create_noise_amplitude_vector_field_df`): the column labeled `(i, j)` is
    placed at `sigma[j, i]` (the non-trivial entry), while the structurally
    zero upper entry `sigma[i, j]` (`i < j`) is left at `0.0` rather than
    mirrored from the same column.

    Parameters
    ----------
    noise_amplitude_dataframe
        Dataframe containing the noise amplitude tensor field data, with
        columns for the mesh grid coordinates and the tensor components.
    column_names
        List of column names corresponding to the features used for computing
        the noise amplitude.

    Returns
    -------
    :
        Tuple of the noise amplitude tensor field, shape `(*grid_shape, ndim,
        ndim)`, and the 1D grid points for each dimension.
    """
    ndim = len(column_names)
    mesh_column_names = [ColumnTemplate.MESH_GRID % name for name in column_names]
    grid_points_1d = [
        np.sort(noise_amplitude_dataframe[column_name].unique())
        for column_name in mesh_column_names
    ]
    grid_shape = tuple(len(points) for points in grid_points_1d)

    sigma = np.full((*grid_shape, ndim, ndim), np.nan)
    for feature_index_1, feature_index_2 in combinations_with_replacement(range(ndim), 2):
        tensor_column_name = ColumnTemplate.NOISE_AMPLITUDE % (
            column_names[feature_index_1],
            column_names[feature_index_2],
        )
        values = noise_amplitude_dataframe[tensor_column_name].to_numpy().reshape(grid_shape)
        sigma[..., feature_index_2, feature_index_1] = values
        if feature_index_1 != feature_index_2:
            sigma[..., feature_index_1, feature_index_2] = 0.0

    return sigma, grid_points_1d
