"""
Methods for converting Kramers-Moyal estimates between stochastic (SDE)
interpretations.

The Kramers-Moyal estimate of the drift as directly calculated corresponds to
the Ito interpretation of the underlying stochastic differential equation. For
an alpha-interpretation SDE

    dx_i = f_i dt + sigma_ij circ_alpha dW_j

the equivalent drift carries an additional (noise-induced) residual term:

    f_i^(alpha) = f_i^(Ito) - alpha * sum_{j,k} sigma_kj * d(sigma_ij)/d(x_k)

where the noise amplitude `sigma` solves `D = (1/2) sigma sigma^T` for the
symmetric Kramers-Moyal diffusion tensor `D`. Setting `alpha = 0` recovers the
Ito interpretation and `alpha = 1/2` the Stratonovich interpretation.
"""

import logging
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import histogramdd
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate


def compute_diffusion_coefficients(
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    bins: list[np.ndarray],
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
    time_step: float,
) -> np.ndarray:
    """
    Compute the diffusion coefficient estimates from data.

    Parameters
    ----------
    dataframe
        DataFrame containing the trajectories.
    column_names
        List of column names corresponding to the trajectory components.
    bins
        List of numpy arrays defining the bin edges for each dimension.
    kernel
        Kramers-Moyal kernel or list of kernels to use for coefficient estimation.
    time_step
        Time step between consecutive trajectory points.

    Returns
    -------
    :
        Array of diffusion coefficients.

    """
    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = get_traj_and_diff(dataframe, column_names)

    # get diffusion tensor estimates in units hours^-1 for each bin in 3D space
    # (Kramers-Moyal coefficient estimation)
    diffusion_coeffs = get_kramers_moyal_coeffs(
        traj_list, d_traj_list, bins=bins, dt=time_step, kernel=kernel
    )[1]

    # Ensure diffusion_coeffs always has a trailing components dimension
    # (shape ..., N) so that downstream functions handle the 1D case
    # (single column) the same as multi-dimensional cases.
    if diffusion_coeffs.ndim == 1:
        diffusion_coeffs = diffusion_coeffs[:, np.newaxis]

    return diffusion_coeffs


def assemble_symmetric_diffusion_tensor(diffusion_coeffs: np.ndarray, ndim: int) -> np.ndarray:
    """Assemble the symmetric diffusion tensor field from Kramers-Moyal components.

    The last axis of `diffusion_coeffs` holds the pure second moments in
    dimension order (`D_00, D_11, ...`) followed by the mixed interaction
    terms in `i < j` order (`D_01, D_02, ..., D_12, ...`), matching the
    output ordering of `get_kramers_moyal_coeffs`.

    Parameters
    ----------
    diffusion_coeffs
        Array of Kramers-Moyal diffusion components with shape
        `(..., ndim * (ndim + 1) // 2)`.
    ndim
        Number of feature dimensions.

    Returns
    -------
    :
        Symmetric diffusion tensor field with shape `(..., ndim, ndim)`.

    """
    n_expected = ndim * (ndim + 1) // 2
    if diffusion_coeffs.shape[-1] != n_expected:
        raise ValueError(
            f"Expected {n_expected} diffusion components for {ndim} dimensions, "
            f"got {diffusion_coeffs.shape[-1]}."
        )

    tensor = np.full((*diffusion_coeffs.shape[:-1], ndim, ndim), np.nan)
    for i in range(ndim):
        tensor[..., i, i] = diffusion_coeffs[..., i]

    component_index = ndim
    for i in range(ndim):
        for j in range(i + 1, ndim):
            tensor[..., i, j] = diffusion_coeffs[..., component_index]
            tensor[..., j, i] = diffusion_coeffs[..., component_index]
            component_index += 1

    return tensor


def compute_noise_amplitude(diffusion_tensor: np.ndarray) -> np.ndarray:
    """
    Solve `D = (1/2) sigma sigma^T` for the noise amplitude at each grid point.

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
    sigma = np.full_like(diffusion_tensor, np.nan)
    n_invalid = 0

    for index in np.ndindex(diffusion_tensor.shape[:-2]):
        tensor = diffusion_tensor[index]
        if not np.all(np.isfinite(tensor)):
            continue
        try:
            sigma[index] = np.linalg.cholesky(2.0 * tensor)
        except np.linalg.LinAlgError:
            n_invalid += 1

    if n_invalid:
        logger.debug(
            "Diffusion tensor was not positive definite at %d grid point(s); "
            "noise amplitude left as NaN there.",
            n_invalid,
        )

    return sigma


def compute_noise_induced_drift_residual(
    sigma: np.ndarray, centers: list[np.ndarray]
) -> np.ndarray:
    """
    Compute the noise-induced drift `sum_{j,k} sigma_kj * d(sigma_ij)/d(x_k)`.

    Parameters
    ----------
    sigma
        Noise amplitude field with shape `(..., ndim, ndim)`.
    centers
        Bin centre arrays for each dimension, used for the physical grid spacing
        of the spatial derivatives.

    Returns
    -------
    :
        Noise-induced drift residual with shape `(..., ndim)`.

    """
    ndim = len(centers)
    spacings = [float(center[1] - center[0]) for center in centers]

    # grad_sigma[..., k, i, j] = d(sigma_ij) / d(x_k)
    grad_sigma = np.stack(
        [np.gradient(sigma, spacings[k], axis=k) for k in range(ndim)],
        axis=-3,
    )

    return np.einsum("...kj,...kij->...i", sigma, grad_sigma)


def compute_residual_magnitude_ratio(
    drift_coeffs: np.ndarray,
    diffusion_coeffs: np.ndarray,
    centers: list[np.ndarray],
) -> np.ndarray:
    """
    Compute the ratio of the magnitude of the noise-induced drift residual to
    the magnitude of the original drift.

    ## Input array shapes

    The input arrays are expected to have the following shapes:

    - `drift_coeffs`: `(..., ndim)`
    - `diffusion_coeffs`: `(..., ndim * (ndim + 1) // 2)`
    - `centers`: list of 1D arrays, each of length corresponding to the number
      of bins in that dimension.

    Parameters
    ----------
    drift_coeffs
        Array of Ito-interpretation drift components.
    diffusion_coeffs
        Array of Kramers-Moyal diffusion tensorcomponents.
    centers
        Bin center arrays for each dimension.

    Returns
    -------
    :
        Ratio of the magnitude of the noise-induced drift residual to the
        magnitude of the original drift, with shape `(..., ndim)`.

    """

    ndim = len(centers)
    diffusion_tensor = assemble_symmetric_diffusion_tensor(diffusion_coeffs, ndim)
    sigma = compute_noise_amplitude(diffusion_tensor)
    noise_induced_drift = compute_noise_induced_drift_residual(sigma, centers)

    ratio = np.linalg.norm(noise_induced_drift, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio /= np.linalg.norm(drift_coeffs, axis=-1)

    return ratio


def compute_binned_residual_covariance(
    positions: np.ndarray,
    residuals: np.ndarray,
    bins: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a histogram-based average of the residual autocovariance,
    `<eta(t) eta(t)^T | x(t) = x>`, conditioned on the state `x(t)`.

    Unlike the kernel-convolution-based Kramers-Moyal diffusion estimate (see
    `compute_diffusion_coefficients`), this averages the outer product of
    already drift-removed residuals (see `compute_centered_residuals`) within
    each bin using a simple sample mean, with no kernel smoothing.

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

    Returns
    -------
    :
        Tuple of the symmetric binned covariance tensor, shape `(*grid_shape,
        ndim, ndim)`, and the number of samples contributing to each bin,
        shape `(*grid_shape,)`.
    """
    ndim = positions.shape[-1]
    is_valid = ~np.isnan(residuals).any(axis=-1) & ~np.isnan(positions).any(axis=-1)
    x_valid = positions[is_valid]
    eta_valid = residuals[is_valid]

    # weight each sample by 1 (for the bin counts) and by the product of
    # residual components for every unique feature pair (for the covariance)
    pair_indices = list(combinations_with_replacement(range(ndim), 2))
    weights = np.stack(
        [np.ones(len(x_valid)), *(eta_valid[:, i] * eta_valid[:, j] for i, j in pair_indices)],
        axis=0,
    )
    hist = histogramdd(x_valid, bins=bins, weights=weights)
    counts, moment_sums = hist[0], hist[1:]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_outer_products = moment_sums / counts

    covariance = np.full((*counts.shape, ndim, ndim), np.nan)
    for (i, j), values in zip(pair_indices, mean_outer_products, strict=True):
        covariance[..., i, j] = values
        covariance[..., j, i] = values

    return covariance, counts


def compute_histogram_noise_amplitude(
    covariance: np.ndarray,
    counts: np.ndarray,
    time_step: float,
    min_samples_per_bin: int = 1,
) -> np.ndarray:
    """
    Convert a histogram-binned residual covariance into the state-dependent
    noise amplitude `sigma(x)`.

    The covariance `<eta eta^T | x>` is first rescaled into the Kramers-Moyal
    diffusion tensor `D(x) = <eta eta^T | x> / (2 dt)`, then factored via
    `compute_noise_amplitude`. Bins with fewer than `min_samples_per_bin`
    contributing samples are masked as NaN before factoring, since the sample
    covariance there is unreliable.

    Parameters
    ----------
    covariance
        Binned residual covariance tensor, shape `(*grid_shape, ndim, ndim)`
        (see `compute_binned_residual_covariance`).
    counts
        Number of samples contributing to each bin, shape `(*grid_shape,)`.
    time_step
        Time step between consecutive observations.
    min_samples_per_bin
        Minimum number of samples required for a bin to be considered valid.

    Returns
    -------
    :
        Noise amplitude field `sigma(x)`, shape `(*grid_shape, ndim, ndim)`.
    """
    diffusion_tensor = covariance / (2.0 * time_step)
    diffusion_tensor[counts < min_samples_per_bin] = np.nan

    return compute_noise_amplitude(diffusion_tensor)


def create_noise_amplitude_vector_field_df(
    noise_amplitude: np.ndarray,
    sample_counts: np.ndarray,
    column_names: list[str | Column.DiffAEData],
    feature_grid: tuple[np.ndarray, ...],
    metadata_dict: dict[str, str | float] | None = None,
) -> pd.DataFrame:
    """
    Create dataframe containing the histogram-based noise amplitude tensor
    field for a single flow condition.

    The output dataframe has columns for the grid points in each dimension
    (matching the layout of `create_drift_vector_field_df`), the unique
    entries of the symmetric noise amplitude tensor `sigma_ij(x)`, and the
    number of residual samples contributing to each grid point.

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
            ..., feature_index_1, feature_index_2
        ].flatten()

    noise_amplitude_dataframe[Column.NoiseAmplitude.SAMPLE_COUNT] = sample_counts.flatten()

    if metadata_dict is not None:
        for key in metadata_dict:
            noise_amplitude_dataframe[key] = metadata_dict[key]

    return noise_amplitude_dataframe
