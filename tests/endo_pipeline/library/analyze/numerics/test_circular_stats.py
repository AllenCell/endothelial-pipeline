import numpy as np
import pytest

from endo_pipeline.library.analyze.numerics.circular_stats import (
    rewrap_polar_angle,
    unwrap_nonsequential_array,
)


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


@pytest.mark.parametrize(
    "wrapped_array, period, expected_unwrapped_array",
    [
        (
            np.array([0.0, np.pi / 2, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 2 * np.pi]),
            2 * np.pi,
            np.array([0.0, np.pi / 2, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0.0]),
        ),
        (
            np.array([1.0, 1.5, -2.5, -2.0, 2.0, 2.5]),
            5.0,
            np.array([1.0, 1.5, 2.5, 3.0, 2.0, 2.5]),
        ),
        (
            np.array([10.0, 12.0, 13.0, 9.0, 16.0]),
            5.0,
            np.array([10.0, 12.0, 8.0, 9.0, 11.0]),
        ),
    ],
)
def test_unwrap_nonsequential_array(wrapped_array, period, expected_unwrapped_array):
    unwrapped_array = unwrap_nonsequential_array(wrapped_array, period)
    np.testing.assert_array_almost_equal(unwrapped_array, expected_unwrapped_array)
