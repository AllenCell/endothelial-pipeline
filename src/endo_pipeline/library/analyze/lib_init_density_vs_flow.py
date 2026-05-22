import logging

import numpy as np

logger = logging.getLogger(__name__)


def test_vector_mean_angle_and_mag():
    test_angles = generate_test_angles()

    for angles in test_angles:
        vec_mean_ang, vec_mean_mag = vector_mean_angle_and_mag(angles)
        print(np.rad2deg(vec_mean_ang), vec_mean_mag)


def vector_mean_angle_and_mag(angles: np.ndarray) -> tuple[float, float]:
    """From a distribution of angles get the vector mean and return its angle and magnitude.
    Input angles must be in radians.
    The returned angle is in the range of [-pi, pi] and the magnitude is in the range of [0, 1].
    """
    # test line below
    xs = np.cos(angles)
    ys = np.sin(angles)

    # the x and y components of the vector mean:
    x_mean = xs.mean()
    y_mean = ys.mean()

    vector_mean_angle = float(np.arctan2(y_mean, x_mean))
    vector_mean_mag = float(np.linalg.norm([x_mean, y_mean]))

    return (vector_mean_angle, vector_mean_mag)


def generate_test_angles() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(1e5)
    # random distribution in the range of [0,180] degrees.
    angles_half_rand = np.deg2rad(np.random.randint(low=0, high=180, size=n_samples))

    # random distribution in the range of [0,360] degrees.
    angles_full_rand = np.deg2rad(np.random.randint(low=0, high=360, size=n_samples))

    # all angles are 45 degrees.
    angles_45 = np.deg2rad(np.linspace(45, 45, n_samples))

    # half of angles are 135 degrees, other half are 315 degrees.
    angles_135 = np.deg2rad(np.linspace(135, 135, n_samples // 2))
    angles_315 = np.deg2rad(np.linspace(315, 315, n_samples // 2))
    angles_bimodal = np.concatenate([angles_135, angles_315])

    return (angles_full_rand, angles_half_rand, angles_45, angles_bimodal)
