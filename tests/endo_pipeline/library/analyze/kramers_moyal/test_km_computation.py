import numpy as np

from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    _get_km_powers,
    get_cartesian_product,
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
