from endo_pipeline.cli import Datasets, PatchType


def main(
    datasets: Datasets | None = None,
    patch_type: PatchType = "grid_based",
    run_low_high_control: bool = False,
    overwrite_results: bool = True,
    fms_upload_only: bool = False,
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

    from endo_pipeline.library.analyze.bandwidth_binwidth_sweep import (
        bootstrap_fixed_points_for_low_high_control,
        generate_flow_field_for_low_high_control,
        get_binwidth_bandwidth_dict,
        get_param_sweep_run_name,
        get_parameter_space,
        upload_bootstrapped_fixed_points_to_fms,
        upload_flow_fields_to_fms,
    )
    from endo_pipeline.workflows.production import bootstrap_fixed_points, generate_flow_field

    logger = logging.getLogger(__name__)

    if datasets is None:
        # use low, the bifurcation intermediate example from figure 4, and high datasets if no
        # datasets are specified by the user
        datasets = ["20250618_20X", "20250319_20X", "20260304_20X", "20250611_20X"]

    if run_low_high_control:
        # when running low-high control, we combine the low and high datasets and only analyze that
        logger.info("Running low-high control with datasets: %s", datasets)
        datasets = ["20250618_20X", "20250611_20X"]

    parameter_space = get_parameter_space()

    if fms_upload_only:
        for bw, kb in tqdm(parameter_space, desc=f"Generating flow fields for: {datasets}"):

            for dataset_name in datasets:
                logger.info("FMS upload only mode enabled.")

                # get tag of parameter scaling combination used for this run
                run_name_for_sweep_condition = get_param_sweep_run_name(bw, kb)

                # Flow field uploads:
                upload_flow_fields_to_fms(
                    sweep_name=run_name_for_sweep_condition,
                    dataset_name=dataset_name,
                    patch_type=patch_type,
                    overwrite_fmsids=overwrite_results,
                )

                # Bootstrapped fixed points uploads:
                upload_bootstrapped_fixed_points_to_fms(
                    sweep_name=run_name_for_sweep_condition,
                    dataset_name=dataset_name,
                    patch_type=patch_type,
                    overwrite_fmsids=overwrite_results,
                )

        # Exit after uploading to FMS and don't run the analyses on the parameters in the sweep.
        return

    for bw, kb in tqdm(parameter_space, desc=f"Generating flow fields for: {datasets}"):
        # apply parameter scaling exponents to the default bin widths and kernel bandwidths
        bin_widths, kernel_bandwidths = get_binwidth_bandwidth_dict(bw, kb)

        # get tag of parameter scaling combination used for this run
        run_name_for_sweep_condition = get_param_sweep_run_name(bw, kb)
        logger.info(
            "Running generate_flow_field for bin width [ %s ], kernel bandwidth [ %s ]",
            bin_widths,
            kernel_bandwidths,
        )

        if run_low_high_control:
            generate_flow_field_for_low_high_control(
                patch_type=patch_type,
                datasets=datasets,
                sweep_name=run_name_for_sweep_condition,
                kernel_bandwidths_dynamics=kernel_bandwidths,
                bin_widths_dynamics=bin_widths,
                overwrite_results=overwrite_results,
            )
            continue

        generate_flow_field.main(
            patch_type=patch_type,
            datasets=datasets,
            sweep_name=run_name_for_sweep_condition,
            kernel_bandwidths_dynamics=kernel_bandwidths,
            bin_widths_dynamics=bin_widths,
            overwrite_results=overwrite_results,
        )

    for bw, kb in tqdm(parameter_space, desc=f"Bootstrapping fixed points for: {datasets}"):
        # apply parameter scaling exponents to the default bin widths and kernel bandwidths
        bin_widths, kernel_bandwidths = get_binwidth_bandwidth_dict(bw, kb)

        # get tag of parameter scaling combination used for this run
        run_name_for_sweep_condition = get_param_sweep_run_name(bw, kb)
        logger.info(
            "Running bootstrap_fixed_points for bin width [ %s ], kernel bandwidth [ %s ]",
            bin_widths,
            kernel_bandwidths,
        )
        if run_low_high_control:
            bootstrap_fixed_points_for_low_high_control(
                patch_type=patch_type,
                datasets=datasets,
                sweep_name=run_name_for_sweep_condition,
                kernel_bandwidths_dynamics=kernel_bandwidths,
                bin_widths_dynamics=bin_widths,
                overwrite_results=overwrite_results,
            )
            continue

        bootstrap_fixed_points.main(
            patch_type=patch_type,
            datasets=datasets,
            sweep_name=run_name_for_sweep_condition,
            kernel_bandwidths_dynamics=kernel_bandwidths,
            bin_widths_dynamics=bin_widths,
            overwrite_results=overwrite_results,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
