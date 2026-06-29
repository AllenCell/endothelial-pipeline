from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None, num_processes: int = 1) -> None:
    """
    Detect and annotate the in-focus z-plane index for each position.

    #quality-control #preprocessing #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe annotate-in-focus-plane -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe annotate-in-focus-plane --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only annotate
    two positions of the first dataset, using the first 10 timepoints.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to annotate.
    num_processes
        Number of processes to use.
    """

    import logging
    from multiprocessing import Pool

    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        load_dataset_config,
        save_dataset_config,
    )
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.process.z_stack_selection import calculate_global_center_plane
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dataset_annotations import IN_FOCUS_PLANE_MANIFEST_NAME

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset, two positions, and 10 timepoints")
        dataset_names = dataset_names[:1]
        max_positions = 2
        max_timepoints = 10
    else:
        max_positions = None
        max_timepoints = None

    # Build dataframe manifest for holding annotation results
    demo_suffix = "_demo" if DEMO_MODE else ""
    manifest_name = f"{IN_FOCUS_PLANE_MANIFEST_NAME}{demo_suffix}"
    manifest = create_dataframe_manifest(manifest_name, workflow_name=__file__)
    save_dataframe_manifest(manifest)

    for dataset_name in dataset_names:
        logger.info("Annotating in focus plane for dataset '%s'", dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        # Get list of valid positions and subset if necessary
        positions = dataset_config.zarr_positions
        if max_positions is not None:
            positions = positions[:max_positions]

        # Parallelize position processing
        args = [(dataset_config, position, max_timepoints) for position in positions]
        with Pool(processes=min(num_processes, len(args))) as pool:
            results = pool.starmap(calculate_global_center_plane, args)

        # Save dataframe to file
        save_path = output_path / f"{dataset_name}_global_center_plane{demo_suffix}.parquet"
        results_df = pd.DataFrame(results)
        results_df.to_parquet(save_path, index=False)

        # Create location object with output path
        location = manifest.locations.get(dataset_name, DataframeLocation())
        location.path = save_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(dataset_config)
            fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None

        # Add dataframe location to dataframe manifest and save
        manifest.locations[dataset_name] = location
        save_dataframe_manifest(manifest)

        if DEMO_MODE:
            logger.info("DEMO_MODE - Will not overwrite dataset config with results")
            continue

        # Update dataset config
        global_center_plane = {
            item[Column.POSITION]: item[Column.Annotations.CENTER_PLANE_MEAN] for item in results
        }
        dataset_config.center_z_plane = global_center_plane
        save_dataset_config(dataset_config)


if __name__ == "__main__":

    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
