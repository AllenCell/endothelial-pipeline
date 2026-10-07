from itertools import product

from endo_pipeline.settings.bandwidth_binwidth_sweep import (
    BINWIDTH_EXPONENT_LIMITS,
    KERNEL_BANDWIDTH_EXPONENT_LIMITS,
    PARAMETER_SCALE_BASE,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.workflow_defaults import DEFAULT_MODEL_RUN_NAME


def get_param_sweep_run_name(bin_scale: int, kernel_scale: int) -> str:
    """
    Generate a name for the parameter sweep based on the bin and kernel scales.

    Parameters
    ----------
    bin_scale
        Exponent used to scale the default bin widths.
    kernel_scale
        Exponent used to scale the default kernel bandwidths.

    Returns
    -------
    :
        Name representing the parameter sweep.
    """
    tag = f"bwScale{bin_scale}_kbScale{kernel_scale}"
    run_name_for_sweep_condition = f"{DEFAULT_MODEL_RUN_NAME}_{tag}"

    return run_name_for_sweep_condition


def apply_parameter_scaling(
    bin_scale: int, kernel_scale: int, param_scale_base: int = PARAMETER_SCALE_BASE
) -> tuple[dict[Column.DiffAEData, float], dict[Column.DiffAEData, float]]:
    """
    Apply scaling exponents to the default bin widths and kernel bandwidths.

    Parameters
    ----------
    bin_scale
        Exponent to scale the default bin widths.
    kernel_scale
        Exponent to scale the default kernel bandwidths.
    param_scale_base
        Base of the exponent for scaling, by default `PARAMETER_SCALE_BASE`

    Returns
    -------
    :
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


def get_parameter_space_scalings() -> list[tuple[int, int]]:
    """
    Get the parameter space of bin width and kernel bandwidth scaling exponents.

    Returns
    -------
    :
        (bin width, kernel bandwidth) combinations of the exponents used for scaling default values.
    """

    # we will scale the default parameters by exponents of base 2
    # (i.e. parameters will be quarter, half, original size, double, quadruple, etc.)
    binwidth_exponent_range: list[int] = list(
        range(BINWIDTH_EXPONENT_LIMITS[0], BINWIDTH_EXPONENT_LIMITS[1] + 1)
    )
    kernel_bandwidth_exponent_range: list[int] = list(
        range(KERNEL_BANDWIDTH_EXPONENT_LIMITS[0], KERNEL_BANDWIDTH_EXPONENT_LIMITS[1] + 1)
    )

    # create the full parameter space of these scaling exponents, which will then be applied to the
    # default bin widths and kernel bandwidths used in the paper
    parameter_space_scaling = list(
        product(binwidth_exponent_range, kernel_bandwidth_exponent_range)
    )

    return parameter_space_scaling
