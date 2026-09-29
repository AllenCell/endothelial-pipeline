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
    from itertools import product

    # from unittest.mock import patch
    from tqdm import tqdm

    # from endo_pipeline.settings import dynamics_workflows, manifest_names
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import DEFAULT_MODEL_RUN_NAME
    from endo_pipeline.workflows.production import bootstrap_fixed_points, generate_flow_field

    logger = logging.getLogger(__name__)

    bin_widths_sweep: list[float] = [1e-3, 0.05, 1.0]
    """Bin widths to sweep over. Applied uniformly to all `DYNAMICS_COLUMN_NAMES`."""

    kernel_bandwidths_sweep: list[float] = [1e-3, 0.15, 1.0]
    """Kernel bandwidths to sweep over. Applied uniformly to all `DYNAMICS_COLUMN_NAMES`."""

    DATASET_LOW_FLOW: str = "20250618_20X"
    """Default dataset used for the sweep (chosen for fast iteration)."""

    if datasets is None:
        datasets = [DATASET_LOW_FLOW]

    parameter_space = list(product(bin_widths_sweep, kernel_bandwidths_sweep))

    for bin_widths, kernel_bandwidths in tqdm(parameter_space):
        tag = f"bw{bin_widths:g}_kb{kernel_bandwidths:g}"
        run_name_for_sweep_condition = f"{DEFAULT_MODEL_RUN_NAME}_{tag}"
        logger.info(
            "Running sweep for bin width [ %s ], kernel bandwidth [ %s ]",
            bin_widths,
            kernel_bandwidths,
        )
        bin_widths_dynamics = dict.fromkeys(DYNAMICS_COLUMN_NAMES, bin_widths)
        kernel_bandwidths_dynamics = dict.fromkeys(DYNAMICS_COLUMN_NAMES, kernel_bandwidths)

        # with (
        #     patch.dict(
        #         dynamics_workflows.BIN_WIDTHS_DYNAMICS,
        #         dict.fromkeys(DYNAMICS_COLUMN_NAMES, bin_widths),
        #     ),
        #     patch.dict(
        #         dynamics_workflows.KERNEL_BANDWIDTHS_DYNAMICS,
        #         dict.fromkeys(DYNAMICS_COLUMN_NAMES, kernel_bandwidths),
        #     ),
        #     patch.dict(
        #         manifest_names.VECTOR_FIELD_MANIFEST_NAMES,
        #         {patch_type: f"{manifest_names.VECTOR_FIELD_MANIFEST_NAMES[patch_type]}_{tag}"},
        #     ),
        #     patch.dict(
        #         manifest_names.FIXED_POINT_MANIFEST_NAMES,
        #         {patch_type: f"{manifest_names.FIXED_POINT_MANIFEST_NAMES[patch_type]}_{tag}"},
        #     ),
        #     patch.dict(
        #         manifest_names.BOOTSTRAPPING_MANIFEST_NAMES,
        #         {patch_type: f"{manifest_names.BOOTSTRAPPING_MANIFEST_NAMES[patch_type]}_{tag}"},
        #     ),
        # ):
        generate_flow_field.main(
            patch_type=patch_type,
            datasets=datasets,
            run_name=run_name_for_sweep_condition,
            kernel_bandwidths_dynamics=kernel_bandwidths_dynamics,
            bin_widths_dynamics=bin_widths_dynamics,
        )
        bootstrap_fixed_points.main(
            patch_type=patch_type,
            datasets=datasets,
            run_name=run_name_for_sweep_condition,
            kernel_bandwidths_dynamics=kernel_bandwidths_dynamics,
            bin_widths_dynamics=bin_widths_dynamics,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
