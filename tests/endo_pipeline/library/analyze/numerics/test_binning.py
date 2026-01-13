import numpy as np

from endo_pipeline.library.analyze.numerics.binning import get_bins


def test_get_bins():
    bin_widths = (0.05, 0.05, 0.05)
    bin_limits = [(0, 1), (-1, 0), (-0.5, 0.5)]
    bin_edges, _ = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)

    for i in range(len(bin_limits)):
        bin_widths_actual = np.diff(bin_edges[i])
        assert np.allclose(bin_widths_actual, bin_widths[i])
