import numpy as np
import pytest

from endo_pipeline.library.analyze.polar_coords import rewrap_polar_angle


@pytest.mark.parametrize(
    "angle, wrapped_range, expected_rewrapped_angle",
    [
        (3 * np.pi / 2, (0, 2 * np.pi), 3 * np.pi / 2),
        (-np.pi / 2, (0, 2 * np.pi), 3 * np.pi / 2),
        (5 * np.pi, (-np.pi, np.pi), -np.pi),
        (-7 * np.pi / 2, (-np.pi, np.pi), np.pi / 2),
        (np.pi / 4, (0, np.pi), np.pi / 4),
        (9 * np.pi / 4, (0, np.pi), np.pi / 4),
        (-3 * np.pi / 4, (0, np.pi), np.pi / 4),
    ],
)
def test_rewrap_polar_angle(angle, wrapped_range, expected_rewrapped_angle):
    rewrapped_angle = rewrap_polar_angle(angle, wrapped_range)
    assert np.isclose(rewrapped_angle, expected_rewrapped_angle)
