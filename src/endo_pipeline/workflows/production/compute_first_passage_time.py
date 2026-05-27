from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    num_processes: int = 1,
) -> None:
    """
    Compute first passage time statistics and parameter sweep.

    #first-passage-time #grid-based #cell-centered

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-first-passage-time -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-first-passage-time --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will compute first
    passage time for the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compute first passage time.
    num_processes
        Number of processes to use.
    """

    import logging
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.analyze.track_integration import (
        compute_first_passage_times_one_dataset,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH
    from endo_pipeline.settings.first_passage_time import (
        FIRST_PASSAGE_TIME_BIN_SIZES,
        FIRST_PASSAGE_TIME_PARAMETER_SWEEP_MANIFEST_NAME,
        FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Set default values
    minimum_track_length = LONG_TRACK_THRESHOLD_LENGTH
    fixed_point_radius_threshold = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE

    # Build dataframe manifest names for results
    statistics_dataframe_manifest = create_dataframe_manifest(
        FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME, workflow_name=__file__
    )
    parameter_sweep_dataframe_manifest = create_dataframe_manifest(
        FIRST_PASSAGE_TIME_PARAMETER_SWEEP_MANIFEST_NAME, workflow_name=__file__
    )

    # Add parameters to dataframe manifests for traceability
    for output_dataframe_manifest in [
        statistics_dataframe_manifest,
        parameter_sweep_dataframe_manifest,
    ]:
        output_dataframe_manifest.parameters = {
            "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
            "run_name": DEFAULT_MODEL_RUN_NAME,
            "fixed_point_radius_threshold": fixed_point_radius_threshold,
            "minimum_track_length": minimum_track_length,
            "bin_sizes": {
                f"{column}": value for column, value in FIRST_PASSAGE_TIME_BIN_SIZES.items()
            },
        }
        save_dataframe_manifest(output_dataframe_manifest)

    # Cap max workers to number of datasets
    num_processes = min(num_processes, len(dataset_names))

    # Bind constant parameters once via partial
    compute_first_passage_times_one_dataset_partial = partial(
        compute_first_passage_times_one_dataset,
        minimum_track_length=minimum_track_length,
        fixed_point_radius_threshold=fixed_point_radius_threshold,
        bin_size_theta_deg=FIRST_PASSAGE_TIME_BIN_SIZES[Column.DiffAEData.POLAR_ANGLE],
        bin_size_radius=FIRST_PASSAGE_TIME_BIN_SIZES[Column.DiffAEData.POLAR_RADIUS],
        bin_size_rho=FIRST_PASSAGE_TIME_BIN_SIZES[Column.DiffAEData.PC3_FLIPPED],
    )

    # Compute first passage times
    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        futures = {
            pool.submit(compute_first_passage_times_one_dataset_partial, dataset_name): dataset_name
            for dataset_name in dataset_names
        }

        for future in tqdm(as_completed(futures), desc="Computing FPT", total=len(futures)):
            results[futures[future]] = future.result()

    # Iterate through each dataset to save results to file
    for dataset_name, (statistics_df_list, parameter_sweep_df_list) in results.items():
        dataset_config = load_dataset_config(dataset_name)

        for manifest, dataframe_list, name_prefix, dataframe_type in [
            (
                statistics_dataframe_manifest,
                statistics_df_list,
                FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME,
                "statistics",
            ),
            (
                parameter_sweep_dataframe_manifest,
                parameter_sweep_df_list,
                FIRST_PASSAGE_TIME_PARAMETER_SWEEP_MANIFEST_NAME,
                "parameters sweep",
            ),
        ]:
            # Save dataframe to file
            dataframe = pd.concat(dataframe_list, ignore_index=True)
            save_path = output_path / f"{name_prefix}_{dataset_name}.parquet"
            dataframe.to_parquet(save_path, index=False)

            # Create location object with output path
            location = manifest.locations.get(dataset_name, DataframeLocation())
            location.path = save_path

            # Upload to FMS (internal only) and replace local path with file id
            if UPLOAD_TO_FMS:
                annotations = build_fms_annotations(
                    dataset_config,
                    model_manifest=load_model_manifest(manifest_name=DEFAULT_MODEL_MANIFEST_NAME),
                    run_name=DEFAULT_MODEL_RUN_NAME,
                    additional_notes=f"First passage time {dataframe_type} dataframe.",
                )
                fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
                location.fmsid = fmsid
                location.path = None

            # Add dataframe location to dataframe manifest and save
            manifest.locations[dataset_name] = location
            save_dataframe_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
