import math

import numpy as np
import pytest

from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    get_smallest_angle_difference,
)


@pytest.mark.parametrize(
    "target_angle,reference_angle,units,expected_diff",
    [
        ([10, 350, 90, -177], [20.1, 10, 271, 178], "deg", [-10.1, -20, 179, 5]),
        ([math.pi, math.pi / 2], [math.pi / 2, math.pi], "rad", [math.pi / 2, -math.pi / 2]),
    ],
)
def test_get_smallest_angle_difference(target_angle, reference_angle, units, expected_diff):
    diff = get_smallest_angle_difference(target_angle, reference_angle, units)

    assert np.allclose(diff, expected_diff)
