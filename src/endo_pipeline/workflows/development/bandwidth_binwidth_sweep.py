from endo_pipeline.cli import Datasets, PatchType


def main(
    datasets: Datasets | None = None,
    patch_type: PatchType = "grid_based",
    overwrite_results: bool = True,
    fms_upload_only: bool = False,
) -> None:
    """
    Sweep bin width and kernel bandwidth parameters for fixed point bootstrapping.

    #dynamical-systems #fixed-points #grid-based #cell-centered #workers

    For each combination of bin width and kernel bandwidth this workflow:

    1. Applies the bin width and kernel bandwidth uniformly across all
       `DYNAMICS_COLUMN_NAMES` and tags the vector field, fixed point, and
       bootstrapping manifest and file names with the parameter combination
       (via `output_file_name_suffix`) so that each combination's outputs are
       kept separate.
    2. Runs `generate_flow_field.main()` to compute the drift vector field and
       baseline fixed points for the parameter combination.
    3. Runs `bootstrap_fixed_points.main()` to compute bootstrap confidence
       intervals for the baseline fixed points found in step 2.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to sweep parameters for.
        Defaults to datasets in `SUPP_FIG_PARAM_SWEEP` which includes low, high,
        bistable intermediate, and monostable intermediate, shear stresses.
    patch_type
        Patch type used to calculate the features.
    overwrite_results
        True to regenerate (or re-upload) outputs that already exist, False to
        skip them.
    fms_upload_only
        True to only upload existing local sweep outputs to FMS without
        running the analyses.
    """

    import logging
    from itertools import product

    from tqdm import tqdm

    from endo_pipeline.library.analyze.bandwidth_binwidth_sweep import (
        get_binwidth_bandwidth_dict,
        get_param_sweep_run_name,
        upload_bootstrapped_fixed_points_to_fms,
        upload_flow_fields_to_fms,
    )
    from endo_pipeline.settings.bandwidth_binwidth_sweep import (
        SWEEP_BINWIDTHS,
        SWEEP_BINWIDTHS_NARROW,
        SWEEP_KERNEL_BANDWIDTHS,
        SWEEP_KERNEL_BANDWIDTHS_NARROW,
    )
    from endo_pipeline.settings.examples import SUPP_FIG_PARAM_SWEEP
    from endo_pipeline.workflows.production import bootstrap_fixed_points, generate_flow_field

    logger = logging.getLogger(__name__)

    if datasets is None:
        # use low, the bifurcation intermediate example from figure 4, and high datasets if no
        # datasets are specified by the user
        datasets = SUPP_FIG_PARAM_SWEEP

    # Get the parameter space of bin widths and kernel bandwidths.
    # combine the broad and narrow parameter sets
    sweep_binwidths_all = list(set(SWEEP_BINWIDTHS) | set(SWEEP_BINWIDTHS_NARROW))
    sweep_kernel_bandwidths_all = list(
        set(SWEEP_KERNEL_BANDWIDTHS) | set(SWEEP_KERNEL_BANDWIDTHS_NARROW)
    )
    # create the full parameter space of all combinations of the bin widths and kernel bandwidths
    parameter_space = list(product(sweep_binwidths_all, sweep_kernel_bandwidths_all))

    if fms_upload_only:
        for bw, kb in tqdm(parameter_space, desc=f"Upload-only mode to FMS for: {datasets}"):

            for dataset_name in datasets:
                logger.info("FMS upload only mode enabled.")

                # get tag of parameter scaling combination used for this run
                run_name_for_sweep_condition = get_param_sweep_run_name(bw, kb)

                # Flow field uploads:
                upload_flow_fields_to_fms(
                    output_file_name_suffix=run_name_for_sweep_condition,
                    dataset_name=dataset_name,
                    patch_type=patch_type,
                    overwrite_fmsids=overwrite_results,
                )

                # Bootstrapped fixed points uploads:
                upload_bootstrapped_fixed_points_to_fms(
                    output_file_name_suffix=run_name_for_sweep_condition,
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

        generate_flow_field.main(
            patch_type=patch_type,
            datasets=datasets,
            output_file_name_suffix=run_name_for_sweep_condition,
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

        bootstrap_fixed_points.main(
            patch_type=patch_type,
            datasets=datasets,
            output_file_name_suffix=run_name_for_sweep_condition,
            kernel_bandwidths_dynamics=kernel_bandwidths,
            bin_widths_dynamics=bin_widths,
            overwrite_results=overwrite_results,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
