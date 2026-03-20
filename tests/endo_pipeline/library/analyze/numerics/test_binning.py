import numpy as np
import pytest

from endo_pipeline.library.analyze.numerics.binning import circpercentile, get_bins


def test_get_bins():
    bin_widths = (0.05, 0.05, 0.05)
    bin_limits = [(0, 1), (-1, 0), (-0.5, 0.5)]
    bin_edges, _ = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)

    for i in range(len(bin_limits)):
        bin_widths_actual = np.diff(bin_edges[i])
        assert np.allclose(bin_widths_actual, bin_widths[i])


def test_get_bins_centers_are_midpoints():
    """Bin centers should be the midpoints between consecutive bin edges."""
    bin_widths = (0.1, 0.2)
    bin_limits = [(0.0, 1.0), (-1.0, 1.0)]
    bin_edges, centers = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)

    for edges, ctrs in zip(bin_edges, centers, strict=True):
        expected_centers = 0.5 * (edges[:-1] + edges[1:])
        assert np.allclose(ctrs, expected_centers)


def test_get_bins_num_bins_matches_ceil():
    """Number of bins should equal ceil((max - min) / width)."""
    bin_width = 0.3
    bin_limits = [(0.0, 1.0)]  # range = 1.0 → ceil(1.0/0.3) = 4
    bin_edges, centers = get_bins(bin_widths=(bin_width,), bin_limits=bin_limits)

    expected_num_bins = int(np.ceil(1.0 / bin_width))  # 4
    assert len(centers[0]) == expected_num_bins
    assert len(bin_edges[0]) == expected_num_bins + 1


def test_get_bins_from_data_covers_data_range():
    """When binning from data, all data points should fall within the bin edges."""
    data = [np.array([[0.0, 0.0], [1.0, 2.0]]), np.array([[0.5, 1.0], [1.5, 2.5]])]
    bin_widths = (0.1, 0.1)
    bin_edges, _ = get_bins(bin_widths=bin_widths, data=data)

    for dim, edges in enumerate(bin_edges):
        all_values = np.concatenate([traj[:, dim] for traj in data])
        assert edges[0] <= all_values.min()
        assert edges[-1] >= all_values.max()


def test_get_bins_from_data_applies_pad():
    """Auto-determined bin limits should extend by `pad` beyond the data extrema."""
    data = [np.array([[0.0, 0.0], [1.0, 2.0]])]
    pad = 0.5
    bin_widths = (0.1, 0.1)
    bin_edges, _ = get_bins(bin_widths=bin_widths, data=data, pad=pad)

    assert bin_edges[0][0] == 0.0 - pad
    assert bin_edges[0][-1] == 1.0 + pad
    assert bin_edges[1][0] == 0.0 - pad
    assert bin_edges[1][-1] == 2.0 + pad


def test_get_bins_from_data_uses_percentile_limits():
    """When percentiles are given, bin limits should be set by the percentile values."""
    data = [np.array([[0.0], [3.0], [4.0], [6.0], [8.0]])]  # 1D data with values from 0 to 8
    lower_p, upper_p = 5.0, 95.0
    bin_widths = (0.1,)
    bin_edges, _ = get_bins(
        bin_widths=bin_widths,
        data=data,
        lower_percentile=lower_p,
        upper_percentile=upper_p,
    )

    expected_min = np.percentile(data[0][:, 0], lower_p)
    expected_max = np.percentile(data[0][:, 0], upper_p)
    assert np.isclose(bin_edges[0][0], expected_min)
    assert np.isclose(bin_edges[0][-1], expected_max)


def test_get_bins_bin_limits_overrides_data():
    """Explicit bin_limits should be used even when data is also provided."""
    # Data spans [0, 10] but bin_limits restricts to [2, 8]; the latter should win.
    data = [np.array([[0.0], [10.0]])]  # shape (2, 1) — 1 dimension
    bin_limits = [(2.0, 8.0)]
    bin_widths = (0.5,)
    bin_edges, _ = get_bins(bin_widths=bin_widths, data=data, bin_limits=bin_limits)

    assert np.isclose(bin_edges[0][0], 2.0)
    assert np.isclose(bin_edges[0][-1], 8.0)


def test_get_bins_1d():
    """Should work correctly for a single dimension."""
    bin_widths = (0.25,)
    bin_limits = [(0.0, 1.0)]
    bin_edges, centers = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)

    assert len(bin_edges) == 1
    assert len(centers) == 1
    assert bin_edges[0][0] == pytest.approx(0.0)
    assert bin_edges[0][-1] == pytest.approx(1.0)


def test_get_bins_negative_range():
    """Should work correctly when the entire range is negative."""
    bin_widths = (0.1,)
    bin_limits = [(-3.0, -1.0)]
    bin_edges, centers = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)

    assert bin_edges[0][0] == pytest.approx(-3.0)
    assert bin_edges[0][-1] == pytest.approx(-1.0)
    assert all(c < 0 for c in centers[0])


def test_get_bins_raises_when_no_data_or_limits():
    """Should raise an error when neither data nor bin_limits is provided."""
    with pytest.raises((ValueError, TypeError)):
        get_bins(bin_widths=(0.1, 0.1))


def test_get_bins_raises_on_dimension_mismatch():
    """Should raise ValueError when bin_widths length doesn't match ndim."""
    bin_limits = [(0.0, 1.0), (0.0, 1.0)]  # 2 dims
    with pytest.raises(ValueError):
        get_bins(bin_widths=(0.1,), bin_limits=bin_limits)  # only 1 width


def test_circpercentile_wraparound_median_degrees():
    angles = np.array([359.0, 1.0, 2.0, 5.0])
    q50 = circpercentile(angles, q=50, polar_range=(0.0, 360.0))
    assert np.isclose(q50, 1.5)


def test_circpercentile_rotation_invariance():
    angles = np.array([350.0, 355.0, 2.0, 8.0])
    q = 75.0
    shift = 37.0
    polar_range = (0.0, 360.0)

    base = circpercentile(angles, q=q, polar_range=polar_range)
    shifted = circpercentile((angles + shift) % 360.0, q=q, polar_range=polar_range)

    assert np.isclose((base + shift) % 360.0, shifted)


def test_circpercentile_matches_np_percentile_when_data_is_contiguous():
    angles = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    q = 25.0
    circular = circpercentile(angles, q=q, polar_range=(0.0, 360.0))
    linear = np.percentile(angles, q=q)
    assert np.isclose(circular, linear)


def test_circpercentile_rewraps_to_signed_pi_range():
    angles = np.array([-3.13, -3.10, 3.11, 3.12])
    q50 = circpercentile(angles, q=50, polar_range=(-np.pi, np.pi))
    q50_from_zero_to_twopi = circpercentile(
        np.mod(angles, 2 * np.pi), q=50, polar_range=(0.0, 2 * np.pi)
    )
    q50_from_zero_to_twopi_rewrapped = ((q50_from_zero_to_twopi + np.pi) % (2 * np.pi)) - np.pi

    assert -np.pi <= q50 < np.pi
    assert np.isclose(q50, q50_from_zero_to_twopi_rewrapped)
