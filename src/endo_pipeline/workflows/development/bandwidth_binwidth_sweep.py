from endo_pipeline.cli import Datasets, PatchType


def main(
    datasets: Datasets | None = None,
    patch_type: PatchType = "grid_based",
) -> None:
    """
    Sweep bin width and kernel bandwidth parameters for fixed point bootstrapping.

    #dynamical-systems #fixed-points #grid-based #cell-centered

    For each combination of bin width and kernel bandwidth in
    `BIN_WIDTHS_SWEEP` x `KERNEL_BANDWIDTHS_SWEEP`, this workflow:

    1. Temporarily overrides `BIN_WIDTHS_DYNAMICS` and `KERNEL_BANDWIDTHS_DYNAMICS`
       (uniformly across all `DYNAMICS_COLUMN_NAMES`) and tags the vector field,
       fixed point, and bootstrapping manifest names with the parameter
       combination so that each combination's outputs are kept separate.
    2. Runs `generate_flow_field.main()` to compute the drift vector field and
       baseline fixed points for the parameter combination.
    3. Runs `bootstrap_fixed_points.main()` to compute bootstrap confidence
       intervals for the baseline fixed points found in step 2.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    datasets
        List of datasets or dataset collections to sweep parameters for.
    """

    import logging

    from tqdm import tqdm

    from endo_pipeline.library.analyze.binwidth_bandwidth_sweep import (
        apply_parameter_scaling,
        get_param_sweep_run_name,
        get_parameter_space_scalings,
    )
    from endo_pipeline.workflows.production import bootstrap_fixed_points, generate_flow_field

    logger = logging.getLogger(__name__)

    if datasets is None:
        # use low, the bifurcation intermediate example from figure 4, and high datasets if no
        # datasets are specified by the user
        datasets = ["20250618_20X", "20250319_20X", "20250611_20X"]

    parameter_space_scaling = get_parameter_space_scalings()

    for bin_scale, kernel_scale in tqdm(parameter_space_scaling, desc="Generating flow fields"):
        # apply parameter scaling exponents to the default bin widths and kernel bandwidths
        bin_widths, kernel_bandwidths = apply_parameter_scaling(bin_scale, kernel_scale)

        # get tag of parameter scaling combination used for this run
        run_name_for_sweep_condition = get_param_sweep_run_name(bin_scale, kernel_scale)
        logger.info(
            "Running generate_flow_field for bin width [ %s ], kernel bandwidth [ %s ]",
            bin_widths,
            kernel_bandwidths,
        )

        generate_flow_field.main(
            patch_type=patch_type,
            datasets=datasets,
            sweep_name=run_name_for_sweep_condition,
            kernel_bandwidths_dynamics=kernel_bandwidths,
            bin_widths_dynamics=bin_widths,
        )

    for bin_scale, kernel_scale in tqdm(parameter_space_scaling, desc="Bootstrapping fixed points"):
        # apply parameter scaling exponents to the default bin widths and kernel bandwidths
        bin_widths, kernel_bandwidths = apply_parameter_scaling(bin_scale, kernel_scale)

        # get tag of parameter scaling combination used for this run
        run_name_for_sweep_condition = get_param_sweep_run_name(bin_scale, kernel_scale)
        logger.info(
            "Running bootstrap_fixed_points for bin width [ %s ], kernel bandwidth [ %s ]",
            bin_widths,
            kernel_bandwidths,
        )
        bootstrap_fixed_points.main(
            patch_type=patch_type,
            datasets=datasets,
            sweep_name=run_name_for_sweep_condition,
            kernel_bandwidths_dynamics=kernel_bandwidths,
            bin_widths_dynamics=bin_widths,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
