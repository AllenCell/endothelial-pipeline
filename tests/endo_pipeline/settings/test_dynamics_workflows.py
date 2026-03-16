import pytest

from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    DYNAMICS_COLUMN_NAMES,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
)


@pytest.mark.parametrize(
    "settings_dict",
    [
        pytest.param(KERNEL_NAMES_DYNAMICS, id="kernel_names"),
        pytest.param(KERNEL_BANDWIDTHS_DYNAMICS, id="kernel_bandwidths"),
        pytest.param(BIN_WIDTHS_DYNAMICS, id="bin_widths"),
    ],
)
def test_kramers_moyal_settings_consistency(settings_dict):
    """
    Test column name consistency across settings dictionaries for dynamics
    workflows.

    This test checks that the column names specified in settings for dynamics
    workflows are consistent across different settings dictionaries, and that
    the corresponding kernels and bin widths can be retrieved without error.
    This is important to ensure that the flow field estimation can be performed
    correctly without key errors, and that the appropriate kernels and bin
    widths are applied for each variable.
    """
    assert all(column_name in settings_dict for column_name in DYNAMICS_COLUMN_NAMES)
