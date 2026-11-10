from typing import Literal

from endo_pipeline.cli import Datasets, PatchType


def main(
    patch_type: PatchType = "grid_based",
    datasets: Datasets | None = None,
    sweep_variable: Literal["start_steady_state", "start_cell_piling"] | None = None,
    sweep_interval_minutes: int = 30,
) -> None:
    """
    Sweep the timepoints used for steady-state analysis and recompute fixed points.

    #dynamical-systems #fixed-points #grid-based #cell-centered #workers

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    datasets
        List of datasets or dataset collections to sweep.
    sweep_variable
        Window edge to sweep: ``start_steady_state`` (lower) or ``start_cell_piling``
        (upper). If omitted, both edges are swept.
    sweep_interval_minutes
        Spacing between swept edge positions, in minutes.
    """

    import logging
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE, NUM_WORKERS, UPLOAD_TO_FMS
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        get_subset_of_timepoint_annotations,
        load_dataset_config,
    )
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_dataframe,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.window_sweep import (
        SweptBoundary,
        build_window_anchors,
        build_window_edge_grid,
        compute_detection_rate,
        compute_single_fit,
        init_window_worker,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.bootstrap_fixed_points import (
        BATCH_SIZE_SCALING_FACTOR,
        BOOTSTRAP_MATCH_RADIUS,
        NUM_BOOTSTRAP_ITERATIONS,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        KERNEL_PERIODS_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        TIME_STEP_IN_MINUTES,
    )
    from endo_pipeline.settings.manifest_names import WINDOW_SWEEP_MANIFEST_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        FEATURES_UNFILTERED_MANIFEST_NAMES,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)
    output_path = get_output_path(__file__)

    max_workers = NUM_WORKERS or 1

    dataset_names = datasets or get_datasets_in_collection("shear_stress")
    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    sweep_interval_timepoints = max(1, round(sweep_interval_minutes / TIME_STEP_IN_MINUTES))

    # Sweep both window edges unless a single one was requested.
    if sweep_variable is not None and sweep_variable not in list(SweptBoundary):
        raise ValueError(f"Unexpected sweep variable: {sweep_variable}")
    boundaries = [sweep_variable] if sweep_variable else list(SweptBoundary)

    column_names = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    manifest_name = FEATURES_UNFILTERED_MANIFEST_NAMES[patch_type]
    feature_manifest = load_dataframe_manifest(manifest_name)

    # Keep not_steady_state (so the steady-state edge can sweep earlier) and
    # cell_piling (so the cell-piling edge can sweep later); still drop genuine
    # artifacts (scope errors, temp artifacts, xy shifts, unfed, etc.).
    timepoint_annotations_to_remove = get_subset_of_timepoint_annotations(
        annotations_to_ignore=[
            TimepointAnnotation.NOT_STEADY_STATE,
            TimepointAnnotation.CELL_PILING,
        ]
    )

    # Kernels and bin widths per column (matches `generate_flow_field`).
    kernels = [
        KramersMoyalKernel(
            name=KERNEL_NAMES_DYNAMICS[column],
            bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column],
            period=KERNEL_PERIODS_DYNAMICS[column],
        )
        for column in column_names
    ]
    bin_widths = [BIN_WIDTHS_DYNAMICS[column] for column in column_names]

    manifest_parameters = {
        "patch_type": patch_type,
        "sweep_interval_minutes": sweep_interval_minutes,
        "sweep_interval_timepoints": sweep_interval_timepoints,
        "num_bootstrap_iterations": NUM_BOOTSTRAP_ITERATIONS,
        "bootstrap_match_radius": BOOTSTRAP_MATCH_RADIUS,
    }
    sweep_manifests = {}
    for boundary in boundaries:
        try:
            manifest = create_dataframe_manifest(
                WINDOW_SWEEP_MANIFEST_NAMES[boundary], workflow_name=__file__
            )
            manifest.parameters = manifest_parameters
            save_dataframe_manifest(manifest)
            sweep_manifests[boundary] = manifest
        except Exception:  # noqa: BLE001
            logger.warning(
                "Could not initialize manifest for boundary '%s' (concurrent access?); "
                "continuing without manifest bookkeeping.",
                boundary,
                exc_info=True,
            )

    for dataset_name in dataset_names:
        if dataset_name not in feature_manifest.locations:
            logger.warning(
                "Dataset '%s' not in manifest '%s'. Skipping.", dataset_name, manifest_name
            )
            continue

        dataset_config = load_dataset_config(dataset_name)

        cell_piling_ranges = (dataset_config.timepoint_annotations or {}).get(
            TimepointAnnotation.CELL_PILING, {}
        )
        position_onsets, position_cell_piling = build_window_anchors(
            dataset_config, cell_piling_ranges
        )

        # Uniform absolute-time grid for the swept edge; identical for both
        # edges so the two sweeps share the same x-axis.
        edge_grid = build_window_edge_grid(dataset_config.duration, sweep_interval_timepoints)
        if not edge_grid:
            logger.warning("Empty edge grid for dataset '%s'. Skipping.", dataset_name)
            continue

        df = load_dataframe(feature_manifest.locations[dataset_name], delay=True)[
            columns_to_compute
        ].compute()

        # Remove artifact-annotated timepoints/positions while retaining
        # not_steady_state and cell_piling rows (see note above).
        df = filter_dataframe_by_annotations(
            df,
            dataset_config,
            timepoint_annotations=timepoint_annotations_to_remove,
        )

        # Per-row window anchors, keyed by position (shared across both edges).
        lower_by_row = df[Column.POSITION].map(position_onsets)
        upper_by_row = df[Column.POSITION].map(position_cell_piling)

        for boundary in boundaries:
            n_flow_conditions = len(dataset_config.flow_conditions)
            n_candidate_windows = len(edge_grid) * n_flow_conditions

            # Work is split into two pool phases so every worker shares in the
            # bootstrap refits for every window, rather than one worker
            # serially paying the full num_bootstrap_iterations cost per edge:
            # first the baseline fit for every (edge, flow_condition) window,
            # then one task per bootstrap iteration for only the windows that
            # found baseline fixed points.
            n_workers = max(1, min(max_workers, n_candidate_windows))
            blas_threads_per_worker = max(1, max_workers // n_workers)

            logger.info(
                "Dataset '%s': sweeping %s over %d edge position(s) [%d, %d] timepoints, "
                "%d flow condition(s), %d bootstrap iteration(s) per window, with %d "
                "worker(s), %d BLAS thread(s) each.",
                dataset_name,
                boundary,
                len(edge_grid),
                edge_grid[0],
                edge_grid[-1],
                n_flow_conditions,
                NUM_BOOTSTRAP_ITERATIONS,
                n_workers,
                blas_threads_per_worker,
            )

            init_args = (
                df,
                lower_by_row,
                upper_by_row,
                boundary,
                dataset_config,
                dataset_name,
                column_names,
                bin_widths,
                kernels,
                blas_threads_per_worker,
                RANDOM_SEED,
            )

            baseline_jobs = [
                (edge, flow_condition_index, None)
                for edge in edge_grid
                for flow_condition_index in range(n_flow_conditions)
            ]
            baseline_batch_size = max(
                1, len(baseline_jobs) // (n_workers * BATCH_SIZE_SCALING_FACTOR)
            )

            sweep_fixed_points: list[pd.DataFrame] = []
            with ProcessPoolExecutor(
                max_workers=n_workers, initializer=init_window_worker, initargs=init_args
            ) as pool:
                baseline_results = list(
                    pool.map(compute_single_fit, baseline_jobs, chunksize=int(baseline_batch_size))
                )

                # Only bootstrap windows with at least one baseline fixed
                # point - nothing to report (or match against) otherwise.
                bootstrap_jobs = [
                    (result.edge_timepoint, result.flow_condition_index, rep)
                    for result in baseline_results
                    if not result.fixed_points.empty
                    for rep in range(NUM_BOOTSTRAP_ITERATIONS)
                ]
                bootstrap_fp_by_window: dict[tuple[int, int], list[pd.DataFrame]] = defaultdict(
                    list
                )
                if bootstrap_jobs:
                    bootstrap_batch_size = max(
                        1, len(bootstrap_jobs) // (n_workers * BATCH_SIZE_SCALING_FACTOR)
                    )
                    for result in pool.map(
                        compute_single_fit, bootstrap_jobs, chunksize=int(bootstrap_batch_size)
                    ):
                        window_key = (result.edge_timepoint, result.flow_condition_index)
                        bootstrap_fp_by_window[window_key].append(result.fixed_points)

            for result in baseline_results:
                if result.fixed_points.empty:
                    continue
                window_key = (result.edge_timepoint, result.flow_condition_index)
                fixed_points_dataframe = result.fixed_points.copy()
                # Left unfiltered here; downstream can threshold on
                # `detection_rate` (paper convention: > 0.4).
                fixed_points_dataframe[Column.FIXED_POINT_DETECTION_RATE] = compute_detection_rate(
                    baseline_fixed_points=fixed_points_dataframe,
                    bootstrap_fixed_points=bootstrap_fp_by_window.get(window_key, []),
                    column_names=column_names,
                    bootstrap_match_radius=BOOTSTRAP_MATCH_RADIUS,
                )
                fixed_points_dataframe[Column.SteadyStateWindow.WINDOW_EDGE_TIMEPOINT] = (
                    result.edge_timepoint
                )
                fixed_points_dataframe[Column.SteadyStateWindow.SWEPT_BOUNDARY] = boundary
                fixed_points_dataframe[Column.SteadyStateWindow.NUM_WINDOW_TIMEPOINTS] = (
                    result.num_window_rows
                )
                sweep_fixed_points.append(fixed_points_dataframe)

            if not sweep_fixed_points:
                logger.warning(
                    "Sweep of %s produced no fixed points for dataset '%s'.",
                    boundary,
                    dataset_name,
                )
                continue

            sweep_results = pd.concat(sweep_fixed_points, ignore_index=True)
            sweep_results[Column.SteadyStateWindow.WINDOW_EDGE_MINUTES] = (
                sweep_results[Column.SteadyStateWindow.WINDOW_EDGE_TIMEPOINT] * TIME_STEP_IN_MINUTES
            )

            save_path = (
                output_path
                / f"window_sweep_{dataset_name}_{boundary}_{sweep_interval_minutes}min.parquet"
            )
            sweep_results.to_parquet(save_path, index=False)
            logger.info(
                "Saved %d fixed point(s) across %d edge position(s) to [ %s ].",
                len(sweep_results),
                sweep_results[Column.SteadyStateWindow.WINDOW_EDGE_TIMEPOINT].nunique(),
                save_path,
            )

            # Upload to FMS (internal only); the file id replaces the local path
            # in the manifest below.
            fmsid = None
            if UPLOAD_TO_FMS:
                annotations = build_fms_annotations(
                    dataset_config,
                    additional_notes=(
                        f"Window sweep of {boundary} fixed points "
                        f"({sweep_interval_minutes} min edge spacing)."
                    ),
                )
                fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")

            # Record the output in the boundary's manifest.
            try:
                manifest = create_dataframe_manifest(
                    WINDOW_SWEEP_MANIFEST_NAMES[boundary], workflow_name=__file__
                )
                manifest.parameters = manifest_parameters
                location = manifest.locations.get(dataset_name, DataframeLocation())
                if fmsid is not None:
                    location.fmsid = fmsid
                    location.path = None
                else:
                    location.path = save_path
                manifest.locations[dataset_name] = location
                save_dataframe_manifest(manifest)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not update manifest for '%s'/%s (concurrent access?); "
                    "parquet saved regardless.",
                    dataset_name,
                    boundary,
                    exc_info=True,
                )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
