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
    from concurrent.futures import ProcessPoolExecutor, as_completed

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
        WINDOW_EDGE_MINUTES,
        WINDOW_EDGE_TIMEPOINT,
        WINDOW_SWEEP_MANIFEST_NAMES,
        build_window_anchors,
        build_window_edge_grid,
        compute_fixed_points_for_edge,
        init_window_worker,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.bootstrap_fixed_points import (
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
    from endo_pipeline.settings.workflow_defaults import (
        FEATURES_UNFILTERED_MANIFEST_NAMES,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)
    output_path = get_output_path(__file__)

    if not NUM_WORKERS:
        NUM_WORKERS = 1

    dataset_names = datasets or get_datasets_in_collection("shear_stress")
    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    sweep_interval_timepoints = max(1, round(sweep_interval_minutes / TIME_STEP_IN_MINUTES))

    # Sweep both window edges unless a single one was requested.
    boundaries = [sweep_variable] if sweep_variable else ["start_steady_state", "start_cell_piling"]

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
            # One worker per edge position (capped by CPUs); spare CPUs -> BLAS threads.
            n_workers = max(1, min(NUM_WORKERS, len(edge_grid)))
            blas_threads_per_worker = max(1, NUM_WORKERS // n_workers)

            logger.info(
                "Dataset '%s': sweeping %s over %d edge position(s) [%d, %d] timepoints "
                "with %d worker(s), %d BLAS thread(s) each.",
                dataset_name,
                boundary,
                len(edge_grid),
                edge_grid[0],
                edge_grid[-1],
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
                NUM_BOOTSTRAP_ITERATIONS,
                BOOTSTRAP_MATCH_RADIUS,
                RANDOM_SEED,
            )

            sweep_fixed_points: list[pd.DataFrame] = []
            with ProcessPoolExecutor(
                max_workers=n_workers, initializer=init_window_worker, initargs=init_args
            ) as pool:
                futures = [pool.submit(compute_fixed_points_for_edge, edge) for edge in edge_grid]
                for future in as_completed(futures):
                    edge, n_rows, fp_dfs = future.result()
                    logger.info("Edge @ %d timepoints: %d window rows.", edge, n_rows)
                    sweep_fixed_points.extend(fp_dfs)

            if not sweep_fixed_points:
                logger.warning(
                    "Sweep of %s produced no fixed points for dataset '%s'.",
                    boundary,
                    dataset_name,
                )
                continue

            sweep_results = pd.concat(sweep_fixed_points, ignore_index=True)
            sweep_results[WINDOW_EDGE_MINUTES] = (
                sweep_results[WINDOW_EDGE_TIMEPOINT] * TIME_STEP_IN_MINUTES
            )

            save_path = (
                output_path
                / f"window_sweep_{dataset_name}_{boundary}_{sweep_interval_minutes}min.parquet"
            )
            sweep_results.to_parquet(save_path, index=False)
            logger.info(
                "Saved %d fixed point(s) across %d edge position(s) to [ %s ].",
                len(sweep_results),
                sweep_results[WINDOW_EDGE_TIMEPOINT].nunique(),
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
