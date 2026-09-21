from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import Datasets, PatchType
from endo_pipeline.settings.bootstrap_fixed_points import (
    BATCH_SIZE_SCALING_FACTOR,
    BOOTSTRAP_MATCH_RADIUS,
    FP_CI_LOWER_PERCENTILE,
    FP_CI_UPPER_PERCENTILE,
    NUM_BOOTSTRAP_ITERATIONS,
    SDE_ALPHA_STRATONOVICH,
)


def main(
    patch_type: PatchType = "grid_based",
    datasets: Datasets | None = None,
    sde_alpha: Annotated[float, Parameter(name="--alpha")] = SDE_ALPHA_STRATONOVICH,
    num_bootstrap_iterations: Annotated[
        int, Parameter(name="--num-iterations")
    ] = NUM_BOOTSTRAP_ITERATIONS,
    bootstrap_match_radius: Annotated[
        float, Parameter(name="--match-dist")
    ] = BOOTSTRAP_MATCH_RADIUS,
    bootstrap_ci_lower_percentile: Annotated[
        float, Parameter(name="--ci-lower")
    ] = FP_CI_LOWER_PERCENTILE,
    bootstrap_ci_upper_percentile: Annotated[
        float, Parameter(name="--ci-upper")
    ] = FP_CI_UPPER_PERCENTILE,
    batch_size_factor: float = BATCH_SIZE_SCALING_FACTOR,
) -> None:
    """
    Identify and bootstrap fixed points of the alpha-corrected drift field.

    #dynamical-systems #fixed-points #grid-based #cell-centered #workers

    The Kramers-Moyal estimate of the drift corresponds to the Ito
    interpretation of the underlying stochastic differential equation. For an
    alpha-interpretation SDE `dx_i = a_i dt + sigma_ij dW_j` the equivalent
    drift carries an additional spurious (noise-induced) term:

    ```
    a_i^(alpha) = a_i^(Ito) - alpha * sum_{j,k} sigma_kj * d(sigma_ij)/d(x_k)
    ```

    where the noise amplitude `sigma` solves `D = (1/2) sigma sigma^T` for the
    symmetric Kramers-Moyal diffusion tensor `D`. Setting `alpha = 0` recovers
    the Ito interpretation and `alpha = 0.5` the Stratonovich interpretation.

    Unlike `bootstrap-fixed-points`, baseline fixed points are recomputed here
    from the corrected drift field rather than loaded from the precomputed
    fixed point manifest, since the correction shifts the fixed point locations.
    Bootstrap confidence intervals and detection rates are then computed with
    the same matching and aggregation scheme as the Ito workflow, so that the
    two sets of results can be compared directly by
    `visualize-alpha-corrected-bootstrap`.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe bootstrap-alpha-corrected-fixed-points -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe bootstrap-alpha-corrected-fixed-points --datasets DATASET_NAME
    ```

    To run the workflow for a different SDE interpretation:

    ```bash
    uv run endopipe bootstrap-alpha-corrected-fixed-points --alpha 1.0
    ```

    ## Parallel processing

    The bootstrap iterations are parallelized across CPU cores based on the
    requested number of worker processes, using the same batching scheme as
    `bootstrap-fixed-points`.

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will perform
    bootstrapping on the first dataset with at most 10 bootstrap iterations.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    datasets
        List of datasets or dataset collections to bootstrap fixed points for.
    sde_alpha
        SDE interpretation parameter used to correct the drift field.
    num_bootstrap_iterations
        Number of bootstrap iterations to perform for each dataset.
    bootstrap_match_radius
        Maximum distance in feature space for a bootstrap fixed point to be
        considered a match to a given baseline fixed point in each iteration.
    bootstrap_ci_lower_percentile
        Percentile defining lower bound of the bootstrap confidence intervals.
    bootstrap_ci_upper_percentile
        Percentile defining upper bound of the bootstrap confidence intervals.
    batch_size_factor
        Factor used to determine size of batch for parallel processing.
    """

    import logging
    import os
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE, NUM_WORKERS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, make_name_unique
    from endo_pipeline.library.analyze.bootstrap_fixed_points import (
        aggregate_bootstrapping_results,
        init_bootstrap_worker,
        match_bootstrap_fixed_points_to_baseline,
        run_flow_field_and_fixed_points,
        run_one_bootstrap_iteration,
        sample_trajectories_and_displacements_for_bootstrapping,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.bootstrap_fixed_points import ALPHA_CORRECTED_MANIFEST_SUFFIX
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        KERNEL_PERIODS_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.flow_field_3d import PAD_BINS_FLOAT
    from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    rng = np.random.default_rng(RANDOM_SEED)

    column_names = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Get feature dataframe manifest for select grid pattern
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Alpha-corrected results are saved under a dedicated manifest so they can be
    # compared against the precomputed Ito-interpretation results.
    name_prefix = f"{BOOTSTRAPPING_MANIFEST_NAMES[patch_type]}_{ALPHA_CORRECTED_MANIFEST_SUFFIX}"
    name_suffix = "_demo" if DEMO_MODE else ""
    bootstrap_results_manifest = create_dataframe_manifest(
        f"{name_prefix}{name_suffix}", workflow_name=__file__
    )

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        logger.warning("DEMO MODE - Limiting bootstrap iterations to <= 10")
        dataset_names = dataset_names[:1]
        num_bootstrap_iterations = min(num_bootstrap_iterations, 10)

    # Initialize kernels and bin widths for each selected column
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    for column_name in column_names:
        kernels.append(
            KramersMoyalKernel(
                name=KERNEL_NAMES_DYNAMICS[column_name],
                bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                period=KERNEL_PERIODS_DYNAMICS[column_name],
            )
        )
        bin_widths.append(BIN_WIDTHS_DYNAMICS[column_name])

    # Add workflow parameters to the output manifest for traceability
    bootstrap_results_manifest.parameters = {
        "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
        "run_name": DEFAULT_MODEL_RUN_NAME,
        "patch_type": patch_type,
        "sde_alpha": sde_alpha,
        "kernels": [
            {
                "column": str(column),
                "name": kernel.name,
                "bandwidth": kernel.bandwidth,
                "period": kernel.period,
            }
            for column, kernel in zip(column_names, kernels, strict=False)
        ],
        "bin_widths": bin_widths,
        "num_bootstrap_iterations": num_bootstrap_iterations,
        "bootstrap_match_radius": bootstrap_match_radius,
    }
    save_dataframe_manifest(bootstrap_results_manifest)

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                feature_dataframe_manifest_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)

        # Load and filter the feature dataframe to steady-state timepoints
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        bootstrap_dataframe_list = []
        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            metadata_dict = {
                Column.DATASET: dataset_name,
                Column.SHEAR_STRESS: shear_stress,
            }

            # Determine bins from the full steady-state data (shared across the
            # baseline and all bootstrap iterations so the fixed-point search
            # uses a consistent grid)
            bins, centers = get_bins(
                bin_widths,
                data=df_flow[column_names].to_numpy(),
                pad=PAD_BINS_FLOAT,
            )

            # Compute trajectories and displacements once from the full
            # steady-state data; the same lists are reused for the baseline and
            # subsampled for each bootstrap iteration.
            trajectories, displacements = get_traj_and_diff(df_flow, column_names)

            # Baseline fixed points of the alpha-corrected drift field
            baseline_fixed_points = run_flow_field_and_fixed_points(
                trajectories=trajectories,
                displacements=displacements,
                df_for_bounds=df_flow,
                bins=bins,
                centers=centers,
                column_names=column_names,
                kernels=kernels,
                metadata_dict=metadata_dict,
                sde_alpha=sde_alpha,
            )

            if baseline_fixed_points.empty:
                logger.warning(
                    "No baseline fixed points for dataset [ %s ], shear [ %s ]. "
                    "Skipping bootstrap for this flow condition.",
                    dataset_name,
                    shear_stress,
                )
                continue

            logger.info(
                "Dataset [ %s ], shear [ %s ]: %d baseline fixed point(s) for alpha = %s.",
                dataset_name,
                shear_stress,
                len(baseline_fixed_points),
                sde_alpha,
            )

            # Generate all resampled trajectory/displacement lists up front, to
            # avoid overhead from repeatedly resampling in each worker process.
            all_sampled_pairs: list[tuple[list[np.ndarray], list[np.ndarray]]] = [
                sample_trajectories_and_displacements_for_bootstrapping(
                    trajectories, displacements, rng=rng
                )
                for _ in range(num_bootstrap_iterations)
            ]

            # Determine worker and per-worker BLAS thread counts that together
            # stay within the 50% of the CPUs allocated by SLURM (or the OS).
            try:
                n_available_cpus = len(os.sched_getaffinity(0))
            except AttributeError:  # Windows
                n_available_cpus = os.cpu_count() or 1

            n_available_cpus = max(1, n_available_cpus // 2)
            n_workers = NUM_WORKERS or n_available_cpus
            blas_threads_per_worker = max(1, n_available_cpus // n_workers)

            # Choose a chunksize that avoids both excessive queue overhead (too
            # small) and uneven load balancing (too large).
            batch_size = max(1, num_bootstrap_iterations // (n_workers * batch_size_factor))
            logger.info(
                "Running %d bootstrap iterations for dataset [ %s ] "
                "with %d worker process(es), %d BLAS thread(s) per worker, batch size %d.",
                num_bootstrap_iterations,
                dataset_name,
                n_workers,
                blas_threads_per_worker,
                batch_size,
            )

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=init_bootstrap_worker,
                initargs=(
                    df_flow,
                    bins,
                    centers,
                    column_names,
                    kernels,
                    blas_threads_per_worker,
                    sde_alpha,
                ),
            ) as executor:
                bootstrap_fixed_points: list[pd.DataFrame] = list(
                    tqdm(
                        executor.map(
                            run_one_bootstrap_iteration, all_sampled_pairs, chunksize=batch_size
                        ),
                        total=num_bootstrap_iterations,
                        desc=f"Bootstrap iterations for dataset: {dataset_name}",
                    )
                )

            n_iterations_with_fpts = sum(len(result_df) > 0 for result_df in bootstrap_fixed_points)
            logger.info(
                "Bootstrap complete for dataset [ %s ]: %d / %d iterations yielded fixed points.",
                dataset_name,
                n_iterations_with_fpts,
                num_bootstrap_iterations,
            )

            # Aggregate bootstrap results by matching fixed points across
            # iterations to the baseline fixed points and computing confidence
            # intervals and detection rates for each baseline fixed point
            matched_coords_flow = match_bootstrap_fixed_points_to_baseline(
                baseline_fixed_points=baseline_fixed_points,
                bootstrap_fixed_points=bootstrap_fixed_points,
                column_names=column_names,
                bootstrap_match_radius=bootstrap_match_radius,
            )
            bootstrap_results_df_flow = aggregate_bootstrapping_results(
                baseline_fixed_points=baseline_fixed_points,
                matched_coords=matched_coords_flow,
                column_names=column_names,
                n_bootstrap=num_bootstrap_iterations,
                bootstrap_ci_lower_percentile=bootstrap_ci_lower_percentile,
                bootstrap_ci_upper_percentile=bootstrap_ci_upper_percentile,
                metadata_dict=metadata_dict,
            )
            bootstrap_dataframe_list.append(bootstrap_results_df_flow)

        if not bootstrap_dataframe_list:
            logger.warning("No bootstrap results for dataset [ %s ]. Skipping save.", dataset_name)
            continue

        # Concatenate results across flow conditions for this dataset
        bootstrap_results_df = pd.concat(bootstrap_dataframe_list, ignore_index=True)

        # Save results locally and update manifest
        output_file_name = f"{name_prefix}_{dataset_name}{name_suffix}.parquet"
        output_save_path = make_name_unique(output_path / output_file_name)
        bootstrap_results_df.to_parquet(output_save_path)
        logger.info(
            "Saved alpha-corrected bootstrap fixed point CI dataframe locally to [ %s ].",
            output_save_path,
        )

        location = bootstrap_results_manifest.locations.get(dataset_name, DataframeLocation())
        location.path = output_save_path
        bootstrap_results_manifest.locations[dataset_name] = location
        save_dataframe_manifest(bootstrap_results_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
