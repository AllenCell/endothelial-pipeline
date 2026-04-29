import pytest

from endo_pipeline.configs.dataset_config import FlowCondition


@pytest.mark.parametrize(
    "shear_stress,expected_shear_stress_bin",
    [
        (4.4, 3),
        (4.5, 6),  # bankers rounding (round half to even) rounds up
        (4.6, 6),
        (5.0, 6),  # bin center - 1
        (6.0, 6),  # bin center
        (7.0, 6),  # bin center + 1
        (7.4, 6),
        (7.5, 6),  # bankers rounding (round half to even) rounds down
        (7.6, 9),
        (8.0, 9),  # bin center - 1
        (9.0, 9),  # bin center
        (10.0, 9),  # bin center + 1
        (10.4, 9),
        (10.5, 12),  # bankers rounding (round half to even) rounds up
        (10.6, 12),
    ],
)
def test_flow_condition_binning(shear_stress, expected_shear_stress_bin):
    flow_condition = FlowCondition(start=0, stop=0, shear_stress=shear_stress)

    assert flow_condition.shear_stress_bin == expected_shear_stress_bin
