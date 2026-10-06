from endo_pipeline.settings.column_names import ColumnName as Column


def apply_parameter_scaling(
    bin_scale: int, kernel_scale: int, param_scale_base: int = 2
) -> tuple[dict[Column.DiffAEData, float], dict[Column.DiffAEData, float]]:
    """
    Apply scaling exponents to the default bin widths and kernel bandwidths.

    Parameters
    ----------
    bin_scale : int
        Exponent to scale the default bin widths.
    kernel_scale : int
        Exponent to scale the default kernel bandwidths.
    param_scale_base : int, optional
        Base of the exponent for scaling, by default 2

    Returns
    -------
    tuple[dict[Column.DiffAEData, float], dict[Column.DiffAEData, float]]
        Scaled bin widths and kernel bandwidths.
    """

    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        KERNEL_BANDWIDTHS_DYNAMICS,
    )

    bin_widths = {
        key: val * param_scale_base**bin_scale for key, val in BIN_WIDTHS_DYNAMICS.items()
    }
    kernel_bandwidths = {
        key: val * param_scale_base**kernel_scale for key, val in KERNEL_BANDWIDTHS_DYNAMICS.items()
    }
    return bin_widths, kernel_bandwidths
