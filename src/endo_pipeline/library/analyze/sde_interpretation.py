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

import numpy as np

logger = logging.getLogger(__name__)


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
