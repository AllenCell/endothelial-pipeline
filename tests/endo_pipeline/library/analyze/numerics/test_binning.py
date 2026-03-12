import numpy as np

from endo_pipeline.library.analyze.numerics.binning import circpercentile, get_bins


def test_get_bins():
    bin_widths = (0.05, 0.05, 0.05)
    bin_limits = [(0, 1), (-1, 0), (-0.5, 0.5)]
    bin_edges, _ = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)

    for i in range(len(bin_limits)):
        bin_widths_actual = np.diff(bin_edges[i])
        assert np.allclose(bin_widths_actual, bin_widths[i])


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
