from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    DYNAMICS_COLUMN_NAMES,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
)


def test_kramers_moyal_settings_consistency():
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

    column_name_checks = []
    for settings_dict in [KERNEL_NAMES_DYNAMICS, KERNEL_BANDWIDTHS_DYNAMICS, BIN_WIDTHS_DYNAMICS]:
        column_name_check = [column_name in settings_dict for column_name in DYNAMICS_COLUMN_NAMES]
        column_name_checks.extend(column_name_check)
    assert all(column_name_checks), (
        f"Column names {DYNAMICS_COLUMN_NAMES} must be present in kernel, bandwidth, "
        "and bin width settings for dynamics workflows."
    )
