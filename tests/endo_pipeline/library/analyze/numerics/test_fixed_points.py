import numpy as np
import pytest

from endo_pipeline.library.analyze.numerics.fixed_points import (
    get_fixed_point_stability,
    is_point_within_percentile_bounds,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel

# Simple non-circular column names used in tests
PC1 = DIFFAE_PC_COLUMN_NAMES[0]
PC3 = DIFFAE_PC_COLUMN_NAMES[2]
THETA = Column.DiffAEData.POLAR_ANGLE

# The polar angle range for wraparound tests
POLAR_RANGE = (0, np.pi)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.0,), True),  # within bounds
        ((-2.0,), False),  # below lower bound
        ((2.0,), False),  # above upper bound
        ((-1.0,), True),  # at lower bound (inclusive)
        ((1.0,), True),  # at upper bound (inclusive)
    ],
)
def test_non_circular_bounds(point, expected):
    lower = {PC1: -1.0}
    upper = {PC1: 1.0}
    assert is_point_within_percentile_bounds(point, [PC1], lower, upper) == expected


@pytest.mark.parametrize(
    "point, expected",
    [
        ((1.0,), True),  # within bounds
        ((0.2,), False),  # below lower bound
        ((2.5,), False),  # above upper bound
        ((0.5,), True),  # at lower bound (inclusive)
        ((2.0,), True),  # at upper bound (inclusive)
    ],
)
def test_circular_no_wraparound_bounds(point, expected):
    """Circular variable where lower <= upper, so no wraparound logic is needed."""
    lower = {THETA: 0.5}
    upper = {THETA: 2.0}
    assert is_point_within_percentile_bounds(point, [THETA], lower, upper) == expected


@pytest.mark.parametrize(
    "point, expected",
    [
        ((2.9,), True),  # near upper end of polar range, in [lower, range_max]
        ((0.1,), True),  # near lower end of polar range, in [range_min, upper]
        ((2.8,), True),  # at lower bound (inclusive)
        ((0.3,), True),  # at upper bound (inclusive)
        ((1.5,), False),  # in the middle of the range, outside both tails
        ((-0.1,), False),  # outside polar_angle_range entirely
    ],
)
def test_circular_with_wraparound_bounds(point, expected):
    """Circular variable where lower > upper, requiring wraparound logic."""
    lower = {THETA: 2.8}
    upper = {THETA: 0.3}
    assert (
        is_point_within_percentile_bounds(
            point, [THETA], lower, upper, polar_angle_range=POLAR_RANGE
        )
        == expected
    )


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.0, 1.0, 2.5), True),  # all within bounds
        ((2.0, 1.0, 2.5), False),  # linear dimension (PC1) out of bounds
        ((0.0, 3.0, 2.5), False),  # circular dimension (THETA) out of bounds
    ],
)
def test_multidimensional_bounds(point, expected):
    columns = [PC1, THETA, PC3]
    lower = {PC1: -1.0, THETA: 0.5, PC3: 0.0}
    upper = {PC1: 1.0, THETA: 2.0, PC3: 5.0}
    assert is_point_within_percentile_bounds(point, columns, lower, upper) == expected


def test_multidimensional_bounds_accepts_numpy_array_point():
    columns = [PC1, THETA, PC3]
    lower = {PC1: -1.0, THETA: 0.5, PC3: 0.0}
    upper = {PC1: 1.0, THETA: 2.0, PC3: 5.0}
    assert is_point_within_percentile_bounds(np.array([0.0, 1.0, 2.5]), columns, lower, upper)


def test_mismatched_point_and_column_names_length_raises():
    with pytest.raises(ValueError, match="does not match number of column names"):
        is_point_within_percentile_bounds(
            (0.0, 1.0),
            [PC1],
            {PC1: -1.0},
            {PC1: 1.0},
        )


@pytest.mark.parametrize(
    "jacobian, expected",
    [
        # All negative real eigenvalues → stable
        (np.diag([-1.0, -2.0]), StabilityLabel.STABLE),
        # All negative real eigenvalues in 3-D → stable
        (np.diag([-0.1, -3.0, -0.5]), StabilityLabel.STABLE),
        # Complex conjugate eigenvalues with negative real part (stable spiral) → stable
        (np.array([[-1.0, -2.0], [2.0, -1.0]]), StabilityLabel.STABLE),
        # All positive real eigenvalues → unstable
        (np.diag([1.0, 2.0]), StabilityLabel.UNSTABLE),
        # All positive real eigenvalues in 3-D → unstable
        (np.diag([0.5, 1.0, 3.0]), StabilityLabel.UNSTABLE),
        # Complex conjugate eigenvalues with positive real part (unstable spiral) → unstable
        (np.array([[1.0, -2.0], [2.0, 1.0]]), StabilityLabel.UNSTABLE),
        # Mixed-sign real eigenvalues → saddle
        (np.diag([-1.0, 1.0]), StabilityLabel.SADDLE),
        # Mixed-sign real eigenvalues in 3-D -> saddle
        (np.diag([-1.0, -0.5, 2.0]), StabilityLabel.SADDLE),
        # All eigenvalues have real part ≈ 0 (pure rotation) → indeterminate
        (np.array([[0.0, -1.0], [1.0, 0.0]]), StabilityLabel.INDETERMINATE),
        # All eigenvalues are zero (zero matrix) → indeterminate
        (np.zeros((2, 2)), StabilityLabel.INDETERMINATE),
    ],
)
def test_get_fixed_point_stability(jacobian, expected):
    assert get_fixed_point_stability(jacobian) == expected
