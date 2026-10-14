import numpy as np

from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    _get_km_powers,
    get_cartesian_product,
    get_kernel_residual_covariance,
    get_kramers_moyal_coeffs,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel


def test_get_km_powers_2d_includes_interaction_term():
    # Density, two drift, two pure diffusion, and one interaction row.
    powers = _get_km_powers(2)

    expected = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [0, 2], [1, 1]])

    assert (powers == expected).all()


def test_diffusion_interaction_term_matches_symmetric_tensor():
    # Every trajectory shares an identical displacement (dx, dy), so the
    # conditional moments equal the exact displacement powers at every populated
    # bin regardless of the kernel. The symmetric Kramers-Moyal diffusion tensor
    # is then D_ij = <dx_i dx_j> / (2 dt), i.e. the interaction term must carry
    # the same 1/2 factor as the diagonal terms.
    dx, dy, dt = 0.3, 0.6, 0.5
    start = np.array([1.1, 1.1])
    displacement = np.array([dx, dy])

    trajectories = [np.stack([start, start + displacement]) for _ in range(400)]
    displacements = [displacement.reshape(1, 2) for _ in range(400)]

    bins = [np.linspace(0.0, 2.0, 9), np.linspace(0.0, 2.0, 9)]
    kernel = KramersMoyalKernel(name="gaussian", bandwidth=0.5)

    _, diffusion = get_kramers_moyal_coeffs(
        trajectories, displacements, bins=bins, dt=dt, kernel=kernel
    )

    # Component order is D_xx, D_yy, D_xy (matches _get_km_powers rows 3-5).
    idx = np.unravel_index(np.nanargmax(diffusion[..., 0]), diffusion[..., 0].shape)
    d_xx, d_yy, d_xy = diffusion[idx]

    assert np.isclose(d_xx, dx**2 / (2 * dt))
    assert np.isclose(d_yy, dy**2 / (2 * dt))
    assert np.isclose(d_xy, dx * dy / (2 * dt))
    # Perfect correlation: off-diagonal equals the geometric mean of the diagonal.
    assert np.isclose(d_xy, np.sqrt(d_xx * d_yy))


def test_kernel_residual_covariance_off_diagonal_is_zero_for_uncorrelated_residuals():
    # Half the samples have residual (ex, ey), half have (ex, -ey), so
    # <eta_x eta_y> = 0 exactly even though neither component is individually
    # zero, while <eta_x^2> and <eta_y^2> are unaffected by the sign flip.
    ex, ey = 0.4, 0.7
    start = np.array([1.1, 1.1])
    positions = np.tile(start, (400, 1))
    residuals = np.array([[ex, ey], [ex, -ey]] * 200)

    bins = [np.linspace(0.0, 2.0, 9), np.linspace(0.0, 2.0, 9)]
    kernel = KramersMoyalKernel(name="gaussian", bandwidth=0.5)

    covariance = get_kernel_residual_covariance(positions, residuals, bins=bins, kernel=kernel)
    idx = np.unravel_index(np.nanargmax(covariance[..., 0, 0]), covariance.shape[:-2])
    tensor = covariance[idx]

    assert np.isclose(tensor[0, 0], ex**2)
    assert np.isclose(tensor[1, 1], ey**2)
    assert np.isclose(tensor[0, 1], 0.0, atol=1e-10)
    assert np.isclose(tensor[1, 0], 0.0, atol=1e-10)


def test_kernel_residual_covariance_off_diagonal_matches_nonzero_cross_moment():
    # Every sample shares the identical residual vector (ex, ey), so
    # <eta_x eta_y> = ex * ey exactly (nonzero), proving the off-diagonal
    # entry is a genuine estimate of the cross moment rather than hardcoded
    # to zero.
    ex, ey = 0.4, 0.7
    start = np.array([1.1, 1.1])
    positions = np.tile(start, (400, 1))
    residuals = np.tile(np.array([ex, ey]), (400, 1))

    bins = [np.linspace(0.0, 2.0, 9), np.linspace(0.0, 2.0, 9)]
    kernel = KramersMoyalKernel(name="gaussian", bandwidth=0.5)

    covariance = get_kernel_residual_covariance(positions, residuals, bins=bins, kernel=kernel)
    idx = np.unravel_index(np.nanargmax(covariance[..., 0, 0]), covariance.shape[:-2])
    tensor = covariance[idx]

    assert np.isclose(tensor[0, 0], ex**2)
    assert np.isclose(tensor[1, 1], ey**2)
    assert np.isclose(tensor[0, 1], ex * ey)
    assert np.isclose(tensor[1, 0], ex * ey)


def test_get_cartesian_product_2d():
    array = [np.array([1, 2, 3]), np.array([4, 5])]
    expected = np.array([[[1, 4], [1, 5]], [[2, 4], [2, 5]], [[3, 4], [3, 5]]])

    product = get_cartesian_product(array)

    assert (product == expected).all()


def test_get_cartesian_product_3d():
    array = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9])]
    expected = np.array(
        [
            [
                [[1, 4, 8], [1, 4, 9]],
                [[1, 5, 8], [1, 5, 9]],
                [[1, 6, 8], [1, 6, 9]],
                [[1, 7, 8], [1, 7, 9]],
            ],
            [
                [[2, 4, 8], [2, 4, 9]],
                [[2, 5, 8], [2, 5, 9]],
                [[2, 6, 8], [2, 6, 9]],
                [[2, 7, 8], [2, 7, 9]],
            ],
            [
                [[3, 4, 8], [3, 4, 9]],
                [[3, 5, 8], [3, 5, 9]],
                [[3, 6, 8], [3, 6, 9]],
                [[3, 7, 8], [3, 7, 9]],
            ],
        ]
    )

    product = get_cartesian_product(array)

    assert (product == expected).all()
